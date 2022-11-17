import functools
from dataclasses import dataclass
from typing import Dict, List

import numpy
import onnx
import onnx.parser
import onnx.shape_inference
import onnxruntime
import pytest

from steelix._arrow import Arrow
from steelix._arrowfields import ArrowFields
from steelix._attributes import AttrFloat32, _Ref
from steelix._function import Function, to_function
from steelix._graph import arguments, results
from steelix._node import OpType
from steelix._type_system import Tensor


@pytest.fixture
def linear(op):
    class LinearFunction(Function):
        @dataclass
        class Attributes:
            slope: AttrFloat32
            shift: AttrFloat32

        class Inputs(ArrowFields):
            X: Arrow

        class Outputs(ArrowFields):
            Y: Arrow

        op_type = OpType("LinearFunction", "steelix.test", 0)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

        def constructor(self, attrs: Dict[str, _Ref], inputs: Inputs) -> Outputs:
            a = op.constant(value_float=attrs["slope"])
            b = op.constant(value_float=attrs["shift"])
            x = inputs.X
            return self.Outputs(op.add(op.mul(a, x), b))

    def linear_inner(x: Arrow, a: float, b: float) -> Arrow:
        return LinearFunction(
            LinearFunction.Attributes(AttrFloat32(a), AttrFloat32(b)),
            LinearFunction.Inputs(x),
        ).outputs.Y

    return linear_inner


@pytest.fixture
def linear2(op, linear):
    class LinearFunction2(Function):
        @dataclass
        class Attributes:
            slope1: AttrFloat32
            shift1: AttrFloat32

        class Inputs(ArrowFields):
            X: Arrow

        class Outputs(ArrowFields):
            Y: Arrow

        op_type = OpType("LinearFunction2", "steelix.test", 0)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

        def constructor(self, attrs: Dict[str, _Ref], inputs: Inputs) -> Outputs:
            return self.Outputs(linear(inputs.X, attrs["slope1"], attrs["shift1"]))

    def linear_inner(x: Arrow, a: float, b: float) -> Arrow:
        return LinearFunction2(
            LinearFunction2.Attributes(AttrFloat32(a), AttrFloat32(b)),
            LinearFunction2.Inputs(x),
        ).outputs.Y

    return linear_inner


@pytest.fixture
def cubic(op, linear):
    class CubicFunction(Function):
        @dataclass
        class Attributes:
            a3: AttrFloat32
            a2: AttrFloat32
            a1: AttrFloat32
            a0: AttrFloat32

        class Inputs(ArrowFields):
            X: Arrow

        class Outputs(ArrowFields):
            Y: Arrow

        op_type = OpType("CubicFunction", "steelix.test.extra", 0)

        attrs: Attributes
        inputs: Inputs
        outputs: Outputs

        def constructor(self, attrs: Dict[str, _Ref], inputs: Inputs) -> Outputs:
            x = inputs.X
            a = op.mul(linear(x, attrs["a3"], attrs["a2"]), op.mul(x, x))
            b = op.add(
                op.mul(x, op.constant(value_float=attrs["a1"])),
                op.constant(value_float=attrs["a0"]),
            )
            y = op.add(a, b)
            return self.Outputs(y)

    def cubic_inner(x: Arrow, a3: float, a2: float, a1: float, a0: float) -> Arrow:
        return CubicFunction(
            CubicFunction.Attributes(
                a3=AttrFloat32(a3),
                a2=AttrFloat32(a2),
                a1=AttrFloat32(a1),
                a0=AttrFloat32(a0),
            ),
            CubicFunction.Inputs(X=x),
        ).outputs.Y

    return cubic_inner


@pytest.fixture
def min_fun_graph(op, linear):
    (start,) = arguments(start=Tensor(numpy.float32, (None,)))
    return results(final=linear(start, 3, 2))


@pytest.fixture
def big_fun_graph(op, linear):
    first, second = arguments(
        first=Tensor(numpy.float32, (None,)), second=Tensor(numpy.float32, (None,))
    )
    return results(final=op.div(linear(op.add(first, second), 3, 2), second))


@pytest.fixture
def double_fun_graph(op, linear):
    (start,) = arguments(start=Tensor(numpy.float32, (None,)))
    return results(final=linear(linear(start, 3, 2), 5, 3))


@pytest.fixture
def wrapped_linear_graph(op, linear2):
    (start,) = arguments(start=Tensor(numpy.float32, (None,)))
    return results(final=linear2(start, 3, 2))


@pytest.fixture
def cubic_graph(op, cubic):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    return results(y=cubic(x, 5, 3, 2, 1))


@pytest.fixture
def cubic_rational_graph(op, cubic):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    return results(y=op.div(cubic(x, 5, 3, 2, 1), cubic(x, 3, 4, 2, 3)))


@pytest.fixture
def cubic_rational_graph_2x3(op, linear, cubic):
    (x,) = arguments(x=Tensor(numpy.float32, (None,)))
    return results(y=linear(op.div(cubic(x, 5, 3, 2, 1), cubic(x, 3, 4, 2, 3)), 2, 3))


@pytest.fixture
def isnan_graph(op):
    x, y, z = arguments(
        x=Tensor(numpy.float64, ()),
        y=Tensor(numpy.float64, ()),
        z=Tensor(numpy.float64, ()),
    )

    @to_function("IsNaN", "steelix.test")
    def isnan(v: Arrow) -> List[Arrow]:
        return [op.not_(op.equal(v, v))]

    return results(
        count=functools.reduce(
            op.add, (op.cast(*isnan(w), to=numpy.int64) for w in (x, y, z))
        )
    )


def test_minimal(onnx_helper, min_fun_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(min_fun_graph, "final", start=a), 3 * a + 2
    )


def test_bigger(onnx_helper, big_fun_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    b = numpy.random.rand(8).astype(numpy.float32) + 1
    onnx_helper.assert_close(
        onnx_helper.run(big_fun_graph, "final", first=a, second=b),
        (3 * (a + b) + 2) / b,
    )


def test_double_application(onnx_helper, double_fun_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(double_fun_graph, "final", start=a), 5 * (3 * a + 2) + 3
    )


def test_onnxruntime_simple_function_support(onnx_helper):
    # See https://github.com/microsoft/onnxruntime/issues/10250 (fixed)
    import onnx.parser

    m = onnx.parser.parse_model(
        """
    <
      ir_version: 8,
      opset_import: [ "" : 14, "local" : 1],
      producer_name: "test",
      producer_version: "1.0",
      model_version: 1,
      doc_string: "Test preprocessing model"
    >
    agraph (uint8[H, W, C] x) => (uint8[H, W, C] x_processed)
    {
        x_processed = local.func(x)
    }

    <
      opset_import: [ "" : 14 ],
      domain: "local",
      doc_string: "function 1"
    >
    f1 (x) => (y) {
        y = Identity(x)
    }

    <
      opset_import: [ "" : 14 ],
      domain: "local",
      doc_string: "function 2"
    >
    f2 (x) => (y) {
        y = Identity(x)
    }

    <
      opset_import: [ "" : 14, "local" : 1 ],
      domain: "local",
      doc_string: "Preprocessing function."
    >
    func (x) => (y) {
        x1 = local.f1(x)
        y = local.f2(x1)
    }

    """
    )

    onnx.checker.check_model(m)
    onnxruntime.InferenceSession(m.SerializeToString())


@pytest.mark.xfail
def test_onnxruntime_nested_function_attr_support():
    m = onnx.parser.parse_model(
        """
    <
      ir_version: 8,
      opset_import: [ "" : 14, "local" : 1],
      model_version: 1
    >
    agraph (int64[H] x) => (int64[H] y)
    {
        y = local.func <outer = 1> (x)
    }

    <
      opset_import: [ "" : 14 ],
      domain: "local",
      doc_string: "function 1"
    >
    f1 <inner> (x) => (y) {
        c = Constant<value_int: int = @inner>()
        y = Add(x, c)
    }

    <
      opset_import: [ "" : 14, "local" : 1 ],
      domain: "local",
      doc_string: "Preprocessing function."
    >
    func <outer> (x) => (y) {
        y = local.f1 <inner: int = @outer> (x)
    }
    """
    )

    onnx.checker.check_model(m)
    session = onnxruntime.InferenceSession(m.SerializeToString())

    x = numpy.array([0, 0, 0], dtype=numpy.int64)
    y = numpy.array([1, 1, 1], dtype=numpy.int64)
    assert (session.run(None, {"x": x})[0] == y).all()


@pytest.mark.skip("ONNXRuntime does not fully support local functions")
def test_minimal_wrapped(onnx_helper, wrapped_linear_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(wrapped_linear_graph, "final", start=a), 3 * a + 2
    )


@pytest.mark.skip("ONNXRuntime does not fully support local functions")
def test_simple_nested_calls_session(onnx_helper, cubic_graph):
    model = cubic_graph.to_onnx_model()
    onnxruntime.InferenceSession(model.SerializeToString())


@pytest.mark.skip("ONNXRuntime does not fully support local functions")
def test_simple_nested_calls(onnx_helper, cubic_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(cubic_graph, "y", x=a),
        (1 + a * (2 + a * (3 + a * 5))),
    )


@pytest.mark.skip("ONNXRuntime does not fully support local functions")
def test_nested_calls(onnx_helper, cubic_rational_graph):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(cubic_rational_graph, "y", x=a),
        (1 + a * (2 + a * (3 + a * 5))) / (3 + a * (2 + a * (4 + a * 3))),
    )


@pytest.mark.skip("ONNXRuntime does not fully support local functions")
def test_complex_nested_calls(onnx_helper, cubic_rational_graph_2x3):
    a = numpy.random.rand(8).astype(numpy.float32)
    onnx_helper.assert_close(
        onnx_helper.run(cubic_rational_graph_2x3, "y", x=a),
        2 * (1 + a * (2 + a * (3 + a * 5))) / (3 + a * (2 + a * (4 + a * 3))) + 3,
    )


def test_to_function_isnan(onnx_helper, isnan_graph):
    nan = numpy.array(numpy.nan, dtype=numpy.float64)
    onnx_helper.assert_close(
        onnx_helper.run(
            isnan_graph,
            "count",
            x=numpy.array(0.0),
            y=numpy.array(1.0),
            z=numpy.array(2.0),
        ),
        0,
    )
    onnx_helper.assert_close(
        onnx_helper.run(isnan_graph, "count", x=nan, y=numpy.array(3.0), z=nan),
        2,
    )
