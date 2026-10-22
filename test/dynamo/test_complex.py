# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


class ComplexDynamoTestCase(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch._functorch.config.enable_complex_wrapper = True

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        torch._functorch.config.enable_complex_wrapper = False


def sample_function(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = torch.mm(a, b)
    return y.abs() + x


class ComplexTests(ComplexDynamoTestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_simple(self):
        fn_c = torch.compile(sample_function, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)

        self.assertEqual(fn_c(a, b, x), sample_function(a, b, x))

    def test_no_complex_in_out(self):
        def f(re, im):
            c = torch.complex(re, im)
            return c.abs()

        re = torch.randn(2, 2, dtype=torch.float32)
        im = torch.randn(2, 2, dtype=torch.float32)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(fn_c(re, im), f(re, im))

    def test_no_complex_inputs(self):
        def f(x, y):
            a = torch.complex(torch.sin(x), torch.cos(y))
            b = torch.complex(torch.cos(x), torch.sin(y))
            return sample_function(a, b, x)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        self.assertEqual(fn_c(x, y), f(x, y))

    def test_list_out(self):
        def f(a, b, x):
            return [sample_function(a, b, x), sample_function(b, a, x)]

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)
        self.assertEqual(fn_c(a, b, x), f(a, b, x))

    def test_incorrect_aliasing_semantics(self):
        def f(a):
            out = torch.view_as_real(a)
            a[...] = torch.zeros_like(a)
            return out

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        ref = f(a)
        test = fn_c(a)
        self.assertEqual(ref, test)

    def test_conj_alias(self):
        def f():
            a = torch.zeros(2, 2, dtype=torch.complex64)
            b = a.conj()
            b[...] = torch.ones_like(b)
            return a

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_neg_alias(self):
        def f():
            a = torch.zeros(2, 2, dtype=torch.complex64)
            b = a._neg_view()
            b[...] = torch.ones_like(b)
            return a

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_complex_output_aliases_input(self):
        def f(a):
            return a[...]

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        ref = f(a)
        test = fn_c(a)
        self.assertEqual(ref, test)
        a[...] = torch.zeros_like(a)
        self.assertEqual(ref, test)

    def test_complex_input_mutation(self):
        def f(a, b):
            a.mul_(b)
            return a.abs()

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        a_ref = a.clone()
        a_test = a.clone()
        self.assertEqual(f(a_test, b), fn_c(a_ref, b))
        self.assertEqual(a_test, a_ref)

    def _check_input_mutation(self, f, backend):
        a = torch.randn(2, 3, dtype=torch.complex64)
        b = torch.randn(2, 3, dtype=torch.complex64)
        a_ref = a.clone()
        a_test = a.clone()
        fn_c = torch.compile(f, backend=backend, fullgraph=True)
        self.assertEqual(fn_c(a_test, b), f(a_ref, b))
        self.assertEqual(a_test, a_ref)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_complex_input_mutation_through_view(self, backend):
        def f(a, b):
            a[0].mul_(b[0])
            return a.abs()

        self._check_input_mutation(f, backend)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_complex_input_mutation_through_view_as_real(self, backend):
        def f(a, b):
            torch.view_as_real(a).mul_(3)
            return a * b

        self._check_input_mutation(f, backend)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_complex_input_mutation_returns_view(self, backend):
        def f(a, b):
            a.mul_(b)
            return a[0]

        self._check_input_mutation(f, backend)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_output_views_other_output(self, backend):
        def f():
            a = torch.randn(5, 5, dtype=torch.complex64)
            return a, a.view(25), torch.view_as_real(a)

        a, c, r = torch.compile(f, backend=backend, fullgraph=True)()
        self.assertEqual(c, a.view(25))
        self.assertEqual(r, torch.view_as_real(a))
        self.assertEqual(c.untyped_storage().data_ptr(), a.untyped_storage().data_ptr())
        self.assertEqual(r.untyped_storage().data_ptr(), a.untyped_storage().data_ptr())
        a.add_(1)
        self.assertEqual(c, a.view(25))
        self.assertEqual(r, torch.view_as_real(a))

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_duplicate_complex_output(self, backend):
        def f():
            a = torch.randn(5, 5, dtype=torch.complex64)
            return a, a

        a, b = torch.compile(f, backend=backend, fullgraph=True)()
        self.assertEqual(a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
        a.add_(1)
        self.assertEqual(a, b)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_conj_output(self, backend):
        def f(x):
            a = x * 2
            return a, a.conj(), (x + 1).conj()

        x = torch.randn(5, dtype=torch.complex64)
        a, c, d = torch.compile(f, backend=backend, fullgraph=True)(x)
        self.assertTrue(c.is_conj())
        self.assertTrue(d.is_conj())
        self.assertEqual((a, c, d), f(x))
        self.assertEqual(c.untyped_storage().data_ptr(), a.untyped_storage().data_ptr())

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_saved_conj_of_complex_input(self, backend):
        def f(a, y):
            return a * y

        a = torch.rand(3, dtype=torch.double, requires_grad=True)
        y = torch.rand(3, dtype=torch.cdouble)
        a_ref = a.detach().clone().requires_grad_()
        out = torch.compile(f, backend=backend, fullgraph=True)(a, y)
        ref = f(a_ref, y)
        self.assertEqual(out, ref)
        out.abs().sum().backward()
        ref.abs().sum().backward()
        self.assertEqual(a.grad, a_ref.grad)

    def test_aliasing_semantics(self):
        def mutate():
            a = torch.ones(2, 2, dtype=torch.complex64)
            out = a[...]
            a[...] = torch.zeros_like(a)
            return out

        mutate_c = torch.compile(mutate, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(mutate(), mutate_c())

    def test_view_as_real(self):
        def f():
            a = torch.ones(2, 2, dtype=torch.complex64)
            b = torch.zeros(2, 2, dtype=torch.complex64)
            return torch.view_as_real(a).clone(), b

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_rope(self):
        """Test the RoPE function, taken from:
        https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L80-L161"""

        def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            t = torch.arange(end, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
            ndim = x.ndim
            assert ndim >= 2  # noqa: S101
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # noqa: S101
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        def apply_rotary_emb(
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        def f(xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            freqs_cis = precompute_freqs_cis(xq.shape[-1], xq.shape[1])
            return apply_rotary_emb(xq, xk, freqs_cis)

        g = torch.Generator().manual_seed(42)
        xq = torch.randn(4, 32, 64, generator=g)
        xk = torch.randn(4, 32, 64, generator=g)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(xq, xk), fn_c(xq, xk))

    @staticmethod
    def _rope_with_cache_gather(x, cache, pos):
        # Llama-style RoPE with the cache rows gathered by position.
        freqs = cache[pos].unsqueeze(1)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * freqs).flatten(-2).type_as(x)

    @staticmethod
    def _rope_inputs(cache_requires_grad):
        g = torch.Generator().manual_seed(0)
        cache = torch.polar(torch.ones(64, 16), torch.randn(64, 16, generator=g))
        pos = torch.randint(0, 64, (16,), generator=g)
        x = torch.randn(16, 4, 32, generator=g, requires_grad=True)
        return x, cache.requires_grad_(cache_requires_grad), pos

    @parametrize("dynamic", [False, True])
    @parametrize("backend", ["aot_eager", "inductor"])
    def test_rope_backward(self, backend, dynamic):
        x, cache, pos = self._rope_inputs(cache_requires_grad=True)
        x_ref, cache_ref = (t.detach().clone().requires_grad_() for t in (x, cache))
        fn_c = torch.compile(
            self._rope_with_cache_gather,
            backend=backend,
            fullgraph=True,
            dynamic=dynamic,
        )
        fn_c(x, cache, pos).sum().backward()
        self._rope_with_cache_gather(x_ref, cache_ref, pos).sum().backward()
        self.assertEqual(x.grad, x_ref.grad)
        self.assertEqual(cache.grad, cache_ref.grad)

    def test_rope_backward_graph_is_real(self):
        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        x, cache, pos = self._rope_inputs(cache_requires_grad=False)
        torch.compile(self._rope_with_cache_gather, backend=backend, fullgraph=True)(
            x, cache, pos
        ).sum().backward()
        (bw,) = backend.bw_graphs
        # Views unpack complex inputs and aten.complex packs complex outputs; any other
        # complex-valued op is complex math left for the eager fallback.
        complex_math = [
            n
            for n in bw.graph.nodes
            if n.op == "call_function"
            and isinstance(n.meta.get("val"), torch.Tensor)
            and n.meta["val"].is_complex()
            and not getattr(n.target, "is_view", False)
            and n.target is not torch.ops.aten.complex.default
        ]
        self.assertEqual(complex_math, [])
        names = [n.name for n in bw.graph.find_nodes(op="placeholder")]
        self.assertTrue(any(name.startswith("tangents_") for name in names), names)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_backward_accumulates_complex_grads(self, backend):
        # Two consumers of one complex intermediate: the forward adds real values,
        # the backward adds the two complex grads of xc.
        def f(x, w1, w2):
            xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(xc * w1).sum() + torch.view_as_real(xc * w2).sum()

        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, 16, generator=g, requires_grad=True)
        w1, w2 = (
            torch.randn(8, 8, dtype=torch.complex64, generator=g) for _ in range(2)
        )
        x_ref = x.detach().clone().requires_grad_()
        torch.compile(f, backend=backend, fullgraph=True)(x, w1, w2).backward()
        f(x_ref, w1, w2).backward()
        self.assertEqual(x.grad, x_ref.grad)


instantiate_parametrized_tests(ComplexTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
