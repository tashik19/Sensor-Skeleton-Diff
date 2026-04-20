import torch
from tqdm import tqdm


class Scheduler:
    def __init__(self, sched_type, T, step, device):
        self.device = device
        t_vals = torch.arange(1, T + 1, step).to(torch.int)

        if sched_type == "cosine":
            def f(t):
                s = 0.008
                return torch.clamp(
                    torch.cos(((t / T + s) / (1 + s)) * (torch.pi / 2)) ** 2 /
                    torch.cos(torch.tensor((s / (1 + s)) * (torch.pi / 2))) ** 2,
                    1e-10, 0.999,
                )
            self.a_bar_t = f(t_vals)
            self.a_bar_t1 = f((t_vals - step).clamp(0))
            self.beta_t = torch.clamp(1 - (self.a_bar_t / self.a_bar_t1), 1e-10, 0.999)
            self.a_t = 1 - self.beta_t
        else:  # linear
            self.beta_t = torch.linspace(1e-4, 0.02, T)[::step]
            self.a_t = 1 - self.beta_t
            self.a_bar_t = torch.stack(
                [torch.prod(self.a_t[:i]) for i in range(1, (T // step) + 1)]
            )
            self.a_bar_t1 = torch.cat([torch.ones(1), self.a_bar_t[:-1]])

        self.sqrt_a_t = torch.sqrt(self.a_t)
        self.sqrt_a_bar_t = torch.sqrt(self.a_bar_t)
        self.sqrt_1_minus_a_bar_t = torch.sqrt(1 - self.a_bar_t)
        self.sqrt_a_bar_t1 = torch.sqrt(self.a_bar_t1)
        self.beta_tilde_t = ((1 - self.a_bar_t1) / (1 - self.a_bar_t)) * self.beta_t

        self._to_device()

    def _to_device(self):
        attrs = [
            "beta_t", "a_t", "a_bar_t", "a_bar_t1",
            "sqrt_a_t", "sqrt_a_bar_t", "sqrt_1_minus_a_bar_t",
            "sqrt_a_bar_t1", "beta_tilde_t",
        ]
        for a in attrs:
            v = getattr(self, a).to(self.device)
            setattr(self, a, v.unsqueeze(-1).unsqueeze(-1))

    def sample_a_t(self, t):           return self.a_t[t - 1]
    def sample_beta_t(self, t):        return self.beta_t[t - 1]
    def sample_a_bar_t(self, t):       return self.a_bar_t[t - 1]
    def sample_a_bar_t1(self, t):      return self.a_bar_t1[t - 1]
    def sample_sqrt_a_t(self, t):      return self.sqrt_a_t[t - 1]
    def sample_sqrt_a_bar_t(self, t):  return self.sqrt_a_bar_t[t - 1]
    def sample_sqrt_1_minus_a_bar_t(self, t): return self.sqrt_1_minus_a_bar_t[t - 1]
    def sample_sqrt_a_bar_t1(self, t): return self.sqrt_a_bar_t1[t - 1]
    def sample_beta_tilde_t(self, t):  return self.beta_tilde_t[t - 1]


class DiffusionProcess:
    def __init__(self, scheduler, device="cpu", ddim_scale=0.0):
        self.scheduler = scheduler
        self.device = device
        self.ddim_scale = ddim_scale

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_ab = self.scheduler.sample_sqrt_a_bar_t(t).to(self.device)
        sqrt_1mab = self.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(self.device)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def _predict_x0(self, xt, context, t, label, model, predict_noise):
        if predict_noise:
            pred_noise = model(xt, context, t, sensor_pred=label)
            sqrt_ab = self.scheduler.sample_sqrt_a_bar_t(t).to(self.device)
            sqrt_1mab = self.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(self.device)
            return (xt - sqrt_1mab * pred_noise) / (sqrt_ab + 1e-8)
        else:
            return model(xt, context, t, sensor_pred=label)

    def _update_ddim(self, x0_pred, xt, t):
        sqrt_ab = self.scheduler.sample_sqrt_a_bar_t(t).to(self.device)
        sqrt_1mab = self.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(self.device)
        eps = (xt - sqrt_ab * x0_pred) / (sqrt_1mab + 1e-8)

        sqrt_ab1 = self.scheduler.sample_sqrt_a_bar_t1(t).to(self.device)
        ab1 = self.scheduler.sample_a_bar_t1(t).to(self.device)
        sqrt_1mab1 = torch.sqrt(1.0 - ab1 + 1e-8)
        return sqrt_ab1 * x0_pred + sqrt_1mab1 * eps

    def _update_ddpm(self, xt, x0_pred, t, step_idx):
        sqrt_ab = self.scheduler.sample_sqrt_a_bar_t(t).to(self.device)
        sqrt_1mab = self.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(self.device)
        eps = (xt - sqrt_ab * x0_pred) / (sqrt_1mab + 1e-8)

        a_t = self.scheduler.sample_a_t(t).to(self.device)
        beta_t = self.scheduler.sample_beta_t(t).to(self.device)
        mean = (1.0 / (torch.sqrt(a_t) + 1e-8)) * (
            xt - (beta_t / (sqrt_1mab + 1e-8)) * eps
        )
        if step_idx > 0:
            beta_tilde = self.scheduler.sample_beta_tilde_t(t).to(self.device)
            return mean + torch.sqrt(beta_tilde + 1e-8) * torch.randn_like(xt)
        return mean

    @torch.no_grad()
    def sample(self, model, context, xt, label, steps, predict_noise=False):
        raw = model.module if hasattr(model, "module") else model

        for step in tqdm(reversed(range(steps)), desc="Sampling", total=steps):
            t = torch.full((xt.size(0),), step + 1, device=self.device, dtype=torch.long)
            x0_pred = self._predict_x0(xt, context, t, label, raw, predict_noise)

            if float(self.ddim_scale) >= 0.999:
                xt = self._update_ddpm(xt, x0_pred, t, step)
            else:
                xt = self._update_ddim(x0_pred, xt, t)

        return xt

    @torch.no_grad()
    def generate(self, model, context, shape, label, steps, predict_noise=False):
        xt = torch.randn(shape, device=self.device)
        return self.sample(model, context, xt, label, steps, predict_noise=predict_noise)
