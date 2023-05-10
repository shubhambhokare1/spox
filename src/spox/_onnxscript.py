# ONNXSCRIPT imports
import onnxscript
from onnxscript.onnx_types import INT64
from onnxscript import opset15 as os_op
from onnxscript import script
from onnxscript.values import OnnxFunction

# ONNX imports
from onnx.helper import get_attribute_value, tensor_dtype_to_np_dtype

# SPOX imports
import spox
from spox._var import Var
import opset.ai.onnx.v17 as op

# Other imports
import numpy as np
import re


def onnx_function_to_spox(
    onnx_function: OnnxFunction
):
    spox_build_inputs = dict()
    spox_build_outputs = dict()
    spox_var_list = dict()

    function_ir = onnx_function.function_ir
    
    # Extract inputs and convert to spox arguments
    function_inputs = function_ir.inputs
    for input in function_inputs:
        input_type = tensor_dtype_to_np_dtype(input.typeinfo.dtype)
        input_shape = input.typeinfo.shape
        spox_build_inputs[str(input)] = spox.argument(
            spox.Tensor(input_type, input_shape)
        )
        spox_var_list[str(input)] = spox_build_inputs[str(input)]
    
    # Traverse ops in OnnxFunction
    function_ops = function_ir.stmts
    for f_op in function_ops:
        operator = op._CONSTRUCTORS[f_op.callee.opname]
        # Find inputs
        args = []
        for arg in f_op.args:
            args.append(spox_var_list[arg])
        # Find attributes
        kwargs = {}
        for attr in f_op.attrs:
            kwargs[attr.attr_proto.name] = get_attribute_value(attr.attr_proto)
        op_output = operator(*args, **kwargs)
        # TODO: Multiple outputs
        op_output._rename(f_op.result[0])
        spox_var_list[f_op.result[0]] = op_output
    
    # TODO: Multiple outputs
    function_outputs = function_ir.outputs
    for output in function_outputs:
        spox_build_outputs['Y'] = spox_var_list[str(output)]

    return spox_build_inputs, spox_build_outputs


@script()
def sample_model(
    A: INT64["N", "M"],
    B: INT64["N", 1],
    X: INT64["M", 1],
):
    ax = os_op.MatMul(A, X)
    summed = os_op.ReduceSum(B + ax)
    empty_shape = os_op.Constant(value_ints=[0])
    result = os_op.Reshape(summed, empty_shape)
    return result


def sample_model_spox():
    a = spox.argument(spox.Tensor(np.int64, ("N", "M")))
    b = spox.argument(spox.Tensor(np.int64, ("N", 1)))
    x = spox.argument(spox.Tensor(np.int64, ("M", 1))) 
    ax = op.matmul(a, x)
    summed = op.reduce_sum(op.add(b, ax))
    empty_shape = op.constant(value=np.array([], dtype=np.int64))
    result = op.reshape(summed, empty_shape)
    onnx_model = spox.build({'a': a, 'b': b, 'x': x}, {'y': result})

def __main__():
    # Print Spox Model
    spox_model = sample_model_spox()
    #print(spox_model)

    # Sample ONNX Model
    os_model = sample_model.to_model_proto()
    #print(os_model)
    
    # Convert OnnxScript function to Spox and build graph
    inputs, outputs = onnx_function_to_spox(sample_model)
    os_to_spox_model = spox.build(inputs, outputs)
    print(os_to_spox_model)
    

__main__()