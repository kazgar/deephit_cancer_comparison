import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepHit(nn.Module):
    def __init__(self, input_dims, network_settings):
        super().__init__()

        self.x_dim = input_dims["x_dim"]

        self.num_Event = input_dims["num_Event"]
        self.num_Category = input_dims["num_Category"]

        self.h_dim_shared = network_settings["h_dim_shared"]
        self.h_dim_CS = network_settings["h_dim_CS"]
        self.num_layers_shared = network_settings["num_layers_shared"]
        self.num_layers_CS = network_settings["num_layers_CS"]
        self.active_fn = network_settings["active_fn"]
        self.initial_W = network_settings["initial_W"]

        self.reg_W = 1e-4
        self.reg_W_out = 1e-4

        self.shared_layers = self.build_shared_layers()
        self.cause_specific_layers = self.build_cause_specific_layers()
        self.output_layer = self.build_output_layer()

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                self.initial_W(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for event_layers in self.cause_specific_layers:
            for layer in event_layers:
                if isinstance(layer, nn.Linear):
                    self.initial_W(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        if isinstance(self.output_layer, nn.Linear):
            self.initial_W(self.output_layer.weight)
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)

    def build_shared_layers(self):
        layers = []
        for i in range(self.num_layers_shared):
            input_dim = self.x_dim if i == 0 else self.h_dim_shared
            layers.append(nn.Linear(input_dim, self.h_dim_shared))
        return nn.ModuleList(layers)

    def build_cause_specific_layers(self):
        layers = []
        for _ in range(self.num_Event):
            event_layers = []
            for i in range(self.num_layers_CS):
                input_dim = self.h_dim_shared if i == 0 else self.h_dim_CS
                event_layers.append(nn.Linear(input_dim, self.h_dim_CS))
            layers.append(nn.ModuleList(event_layers))
        return nn.ModuleList(layers)

    def build_output_layer(self):
        return nn.Linear(self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category)

    def forward(self, x):
        for layer in self.shared_layers:
            x = self.active_fn(layer(x))

        outputs = []
        for event_layers in self.cause_specific_layers:
            h = x
            for layer in event_layers:
                h = self.active_fn(layer(h))
            outputs.append(h)

        out = torch.stack(outputs, dim=1)
        out = out.view(out.size(0), -1)

        out = F.dropout(out, p=0.4, training=self.training)

        out = self.output_layer(out)
        out = out.view(-1, self.num_Event, self.num_Category)
        return F.softmax(out, dim=-1)

    def loss_log_likelihood(self, k_mb, m1_mb, predictions):
        I_1 = torch.sign(k_mb)
        tmp1 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1)
        tmp2 = torch.sum(torch.sum(m1_mb * predictions, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2)
        return -torch.mean(tmp1 + tmp2)

    def loss_ranking(self, t_mb, k_mb, m2_mb, predictions):
        sigma1 = torch.tensor(0.1, dtype=torch.float32, device=predictions.device)
        eta = []

        for e in range(self.num_Event):
            one_vector = torch.ones_like(t_mb, dtype=torch.float32)

            I_2 = (k_mb == (e + 1)).float()
            I_2_diag = torch.diag(I_2.squeeze())

            tmp_e = predictions[:, e, :]

            R = torch.matmul(tmp_e, m2_mb.T)
            diag_R = torch.diag(R)
            R = torch.matmul(one_vector, diag_R.unsqueeze(0)) - R
            R = R.T

            T = torch.nn.functional.relu(
                torch.sign(torch.matmul(one_vector, t_mb.T) - torch.matmul(t_mb, one_vector.T))
            )
            T = torch.matmul(I_2_diag, T)

            exp_term = torch.exp(-R / sigma1)

            tmp_eta = torch.mean(T * exp_term, dim=1, keepdim=True)
            eta.append(tmp_eta)

        eta = torch.stack(eta, dim=1)
        eta = torch.mean(eta.view(-1, self.num_Event), dim=1, keepdim=True)
        loss = torch.sum(eta)
        return loss

    def loss_calibration(self, k_mb, m2_mb, predictions):
        eta_calibration = []
        for e in range(self.num_Event):
            I_2 = (k_mb == (e + 1)).float()
            tmp_e = predictions[:, e, :]
            r = torch.sum(tmp_e * m2_mb, dim=1)
            tmp_eta = torch.mean((r - I_2) ** 2, dim=0, keepdim=True)
            eta_calibration.append(tmp_eta)
        eta_calibration = torch.stack(eta_calibration, dim=1)
        eta_calibration = torch.mean(eta_calibration.view(-1, self.num_Event), dim=1, keepdim=True)
        return torch.sum(eta_calibration)

    def compute_loss(self, DATA, MASK, PARAMETERS, predictions):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK
        alpha, beta, gamma = PARAMETERS

        loss1 = self.loss_log_likelihood(k_mb, m1_mb, predictions)
        loss2 = self.loss_ranking(t_mb, k_mb, m2_mb, predictions)
        loss3 = self.loss_calibration(k_mb, m2_mb, predictions)
        total_loss = alpha * loss1 + beta * loss2 + gamma * loss3

        l2_reg_loss = torch.tensor(0.0, device=predictions.device)
        l1_reg_loss = torch.tensor(0.0, device=predictions.device)

        for layer in self.shared_layers:
            for param in layer.parameters():
                if param.requires_grad:
                    l2_reg_loss += torch.sum(param**2)

        for event_layers in self.cause_specific_layers:
            for layer in event_layers:
                for param in layer.parameters():
                    if param.requires_grad:
                        l2_reg_loss += torch.sum(param**2)

        if self.output_layer.weight.requires_grad:
            l1_reg_loss += torch.sum(torch.abs(self.output_layer.weight))

        total_loss += self.reg_W * l2_reg_loss + self.reg_W_out * l1_reg_loss

        return total_loss

    def training_step(self, DATA, MASK, PARAMETERS, optimizer):
        x_mb, k_mb, t_mb = DATA
        m1_mb, m2_mb = MASK

        optimizer.zero_grad()

        predictions = self(x_mb)

        loss = self.compute_loss(DATA, MASK, PARAMETERS, predictions)

        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            return self.forward(x_test)
