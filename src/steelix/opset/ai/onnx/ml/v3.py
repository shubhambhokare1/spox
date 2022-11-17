# flake8: noqa
import typing  # noqa: F401
from dataclasses import dataclass
from typing import (  # noqa: F401
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing import cast as typing_cast  # noqa: F401

import numpy as np  # noqa: F401

from spox._arrow import Arrow, _nil, result_type  # noqa: F401
from spox._arrowfields import ArrowFields, NoArrows  # noqa: F401
from spox._attributes import (
    AttrDtype,
    AttrFloat32,
    AttrFloat32s,
    AttrGraph,
    AttrInt64,
    AttrInt64s,
    AttrString,
    AttrStrings,
    AttrTensor,
    AttrType,
)
from spox._graph import Graph, subgraph  # noqa: F401
from spox._internal_op import intro  # noqa: F401
from spox._node import OpType  # noqa: F401
from spox._standard import InferenceError, StandardNode  # noqa: F401
from spox._type_system import Sequence as SpoxSequence  # noqa: F401
from spox._type_system import Tensor, Type, type_match


class _ArrayFeatureExtractor(StandardNode):
    @dataclass
    class Attributes:
        pass

    class Inputs(ArrowFields):
        X: Arrow
        Y: Arrow

    class Outputs(ArrowFields):
        Z: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if not self.inputs.fully_typed:
            return {}
        xt, yt = self.inputs.X.unwrap_tensor(), self.inputs.Y.unwrap_tensor()
        assert xt.shape is not None  # already checked with fully_typed
        assert yt.shape is not None  # already checked with fully_typed
        if len(xt.shape) < 1:
            raise InferenceError("Expected rank >= 1")
        if len(yt.shape) != 1:
            raise InferenceError("Input `Y` must be of rank 1.")
        shape = tuple(list(xt.shape[:-1]) + [yt.shape[-1]])  # type: ignore
        return {"Z": Tensor(xt.dtype, shape)}

    op_type = OpType("ArrayFeatureExtractor", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Binarizer(StandardNode):
    @dataclass
    class Attributes:
        threshold: AttrFloat32

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        return {"Y": self.inputs.X.type} if self.inputs.X.type is not None else {}

    op_type = OpType("Binarizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CastMap(StandardNode):
    @dataclass
    class Attributes:
        cast_to: AttrString
        map_form: AttrString
        max_map: AttrInt64

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("CastMap", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _CategoryMapper(StandardNode):
    @dataclass
    class Attributes:
        cats_int64s: Optional[AttrInt64s]
        cats_strings: Optional[AttrStrings]
        default_int64: AttrInt64
        default_string: AttrString

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if not self.inputs.fully_typed:
            return {}
        cats1, cats2 = self.attrs.cats_int64s, self.attrs.cats_strings
        if cats1 is None or cats2 is None:
            raise InferenceError("Missing required attributes.")
        if len(cats1.value) != len(cats2.value):
            raise InferenceError("Categories lists have mismatched lengths.")
        t = self.inputs.X.unwrap_tensor()
        (elem_type,) = {np.int64, np.str_} - {t.dtype.type}  # type: ignore
        return {"Y": Tensor(elem_type, t.shape)}

    op_type = OpType("CategoryMapper", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _DictVectorizer(StandardNode):
    @dataclass
    class Attributes:
        int64_vocabulary: Optional[AttrInt64s]
        string_vocabulary: Optional[AttrStrings]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("DictVectorizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _FeatureVectorizer(StandardNode):
    @dataclass
    class Attributes:
        inputdimensions: Optional[AttrInt64s]

    class Inputs(ArrowFields):
        X: Sequence[Arrow]

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("FeatureVectorizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Imputer(StandardNode):
    @dataclass
    class Attributes:
        imputed_value_floats: Optional[AttrFloat32s]
        imputed_value_int64s: Optional[AttrInt64s]
        replaced_value_float: AttrFloat32
        replaced_value_int64: AttrInt64

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if not self.inputs.fully_typed:
            return {}
        t = self.inputs.X.unwrap_tensor()
        # We verify if the attributes are set correctly and matching the input elem type
        cases = {
            np.int64: (
                self.attrs.imputed_value_int64s,
                self.attrs.replaced_value_int64,
            ),
            np.float32: (
                self.attrs.imputed_value_floats,
                self.attrs.replaced_value_float,
            ),
        }
        for key, (imp, rep) in cases.items():
            if t.dtype.type is key:
                if not all(
                    imp1 is None for key1, (imp1, rep1) in cases.items() if key != key1
                ):
                    raise InferenceError("Only one input imputed type may be set.")
                break
        else:
            raise InferenceError("No matching element type")
        if imp is None:
            raise InferenceError("Value list for imputation is required.")
        # If the number of features is known (last row, we can check this here)
        sim = t.shape
        last = sim[-1] if sim else 1
        if isinstance(last, int) and len(imp.value) not in {1, last}:
            raise InferenceError(
                f"Mismatched expected ({len(imp.value)}) and actual ({last}) feature count."
            )
        return {"Y": t}

    op_type = OpType("Imputer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LabelEncoder(StandardNode):
    @dataclass
    class Attributes:
        default_float: AttrFloat32
        default_int64: AttrInt64
        default_string: AttrString
        keys_floats: Optional[AttrFloat32s]
        keys_int64s: Optional[AttrInt64s]
        keys_strings: Optional[AttrStrings]
        values_floats: Optional[AttrFloat32s]
        values_int64s: Optional[AttrInt64s]
        values_strings: Optional[AttrStrings]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("LabelEncoder", "ai.onnx.ml", 2)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LinearClassifier(StandardNode):
    @dataclass
    class Attributes:
        classlabels_ints: Optional[AttrInt64s]
        classlabels_strings: Optional[AttrStrings]
        coefficients: AttrFloat32s
        intercepts: Optional[AttrFloat32s]
        multi_class: AttrInt64
        post_transform: AttrString

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow
        Z: Arrow

    op_type = OpType("LinearClassifier", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _LinearRegressor(StandardNode):
    @dataclass
    class Attributes:
        coefficients: Optional[AttrFloat32s]
        intercepts: Optional[AttrFloat32s]
        post_transform: AttrString
        targets: AttrInt64

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if not self.inputs.fully_typed:
            return {}
        sim = self.inputs.X.unwrap_tensor().shape
        assert sim is not None
        if len(sim) == 2:
            return {"Y": Tensor(np.float32, sim)}
        elif len(sim) == 1:
            return {"Y": Tensor(np.float32, (1, sim[0]))}
        elif len(sim) == 0:
            return {"Y": Tensor(np.float32, (1, 1))}
        else:
            raise InferenceError("Input shape must be at most a matrix.")

    op_type = OpType("LinearRegressor", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Normalizer(StandardNode):
    @dataclass
    class Attributes:
        norm: AttrString

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if self.attrs.norm.value not in ("MAX", "L1", "L2"):
            raise InferenceError(
                f"Unknown normalisation method `{self.attrs.norm.value}`"
            )
        return {"Y": self.inputs.X.type} if self.inputs.X.type is not None else {}

    op_type = OpType("Normalizer", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _OneHotEncoder(StandardNode):
    @dataclass
    class Attributes:
        cats_int64s: Optional[AttrInt64s]
        cats_strings: Optional[AttrStrings]
        zeros: AttrInt64

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if not self.inputs.fully_typed:
            return {}
        if self.attrs.cats_int64s:
            n_encodings = len(self.attrs.cats_int64s.value)
        elif self.attrs.cats_strings:
            n_encodings = len(self.attrs.cats_strings.value)
        else:
            raise InferenceError(
                "Either `cats_int64s` or `cats_strings` attributes must be set."
            )
        shape = (*self.inputs.X.unwrap_tensor().shape, n_encodings)  # type: ignore
        return {"Y": Tensor(dtype=np.float32, shape=shape)}

    op_type = OpType("OneHotEncoder", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SVMClassifier(StandardNode):
    @dataclass
    class Attributes:
        classlabels_ints: Optional[AttrInt64s]
        classlabels_strings: Optional[AttrStrings]
        coefficients: Optional[AttrFloat32s]
        kernel_params: Optional[AttrFloat32s]
        kernel_type: AttrString
        post_transform: AttrString
        prob_a: Optional[AttrFloat32s]
        prob_b: Optional[AttrFloat32s]
        rho: Optional[AttrFloat32s]
        support_vectors: Optional[AttrFloat32s]
        vectors_per_class: Optional[AttrInt64s]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow
        Z: Arrow

    op_type = OpType("SVMClassifier", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _SVMRegressor(StandardNode):
    @dataclass
    class Attributes:
        coefficients: Optional[AttrFloat32s]
        kernel_params: Optional[AttrFloat32s]
        kernel_type: AttrString
        n_supports: AttrInt64
        one_class: AttrInt64
        post_transform: AttrString
        rho: Optional[AttrFloat32s]
        support_vectors: Optional[AttrFloat32s]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    op_type = OpType("SVMRegressor", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _Scaler(StandardNode):
    @dataclass
    class Attributes:
        offset: Optional[AttrFloat32s]
        scale: Optional[AttrFloat32s]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if self.inputs.X.type is None:
            return {}
        sc, off = self.attrs.scale, self.attrs.offset
        if sc is None or off is None:
            raise InferenceError("Scale and offset are required attributes.")
        t = self.inputs.X.unwrap_tensor()
        # If the number of features is known (last row, we can check this here)
        last = t.shape[-1] if t.shape else 1
        if isinstance(last, int) and len(sc.value) not in {1, last}:
            raise InferenceError(
                f"Mismatched expected ({len(sc.value)}) and actual ({last}) feature count for scale."
            )
        if isinstance(last, int) and len(off.value) not in {1, last}:
            raise InferenceError(
                f"Mismatched expected ({len(off.value)}) and actual ({last}) feature count for offset."
            )
        return {"Y": Tensor(np.float32, t.shape)}

    op_type = OpType("Scaler", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _TreeEnsembleClassifier(StandardNode):
    @dataclass
    class Attributes:
        base_values: Optional[AttrFloat32s]
        base_values_as_tensor: Optional[AttrTensor]
        class_ids: Optional[AttrInt64s]
        class_nodeids: Optional[AttrInt64s]
        class_treeids: Optional[AttrInt64s]
        class_weights: Optional[AttrFloat32s]
        class_weights_as_tensor: Optional[AttrTensor]
        classlabels_int64s: Optional[AttrInt64s]
        classlabels_strings: Optional[AttrStrings]
        nodes_falsenodeids: Optional[AttrInt64s]
        nodes_featureids: Optional[AttrInt64s]
        nodes_hitrates: Optional[AttrFloat32s]
        nodes_hitrates_as_tensor: Optional[AttrTensor]
        nodes_missing_value_tracks_true: Optional[AttrInt64s]
        nodes_modes: Optional[AttrStrings]
        nodes_nodeids: Optional[AttrInt64s]
        nodes_treeids: Optional[AttrInt64s]
        nodes_truenodeids: Optional[AttrInt64s]
        nodes_values: Optional[AttrFloat32s]
        nodes_values_as_tensor: Optional[AttrTensor]
        post_transform: AttrString

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow
        Z: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        e = (
            len(self.attrs.class_ids.value)
            if self.attrs.class_ids is not None
            else None
        )
        if self.attrs.classlabels_strings is not None:
            y_type = np.str_
        elif self.attrs.classlabels_int64s is not None:
            y_type = np.int64  # type: ignore
        else:
            raise InferenceError(
                "Either string or int64 class labels should be defined"
            )
        if self.inputs.fully_typed:
            shape = self.inputs.X.unwrap_tensor().shape
            assert shape is not None  # already checked with fully_typed
            if len(shape) != 2:
                raise InferenceError("Expected input to be a matrix.")
            n = shape[0]
        else:
            n = None
        return {"Y": Tensor(y_type, (n,)), "Z": Tensor(np.float32, (n, e))}

    op_type = OpType("TreeEnsembleClassifier", "ai.onnx.ml", 3)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _TreeEnsembleRegressor(StandardNode):
    @dataclass
    class Attributes:
        aggregate_function: AttrString
        base_values: Optional[AttrFloat32s]
        base_values_as_tensor: Optional[AttrTensor]
        n_targets: Optional[AttrInt64]
        nodes_falsenodeids: Optional[AttrInt64s]
        nodes_featureids: Optional[AttrInt64s]
        nodes_hitrates: Optional[AttrFloat32s]
        nodes_hitrates_as_tensor: Optional[AttrTensor]
        nodes_missing_value_tracks_true: Optional[AttrInt64s]
        nodes_modes: Optional[AttrStrings]
        nodes_nodeids: Optional[AttrInt64s]
        nodes_treeids: Optional[AttrInt64s]
        nodes_truenodeids: Optional[AttrInt64s]
        nodes_values: Optional[AttrFloat32s]
        nodes_values_as_tensor: Optional[AttrTensor]
        post_transform: AttrString
        target_ids: Optional[AttrInt64s]
        target_nodeids: Optional[AttrInt64s]
        target_treeids: Optional[AttrInt64s]
        target_weights: Optional[AttrFloat32s]
        target_weights_as_tensor: Optional[AttrTensor]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Y: Arrow

    def infer_output_types(self) -> Dict[str, Type]:
        if self.inputs.fully_typed:
            shape = self.inputs.X.unwrap_tensor().shape
            assert shape is not None  # already checked with fully_typed
            if len(shape) != 2:
                raise InferenceError("Expected input to be a matrix.")
            assert shape is not None
            n = shape[0]
        else:
            n = None
        e = self.attrs.n_targets.value if self.attrs.n_targets is not None else None
        return {"Y": Tensor(np.float32, (n, e))}

    op_type = OpType("TreeEnsembleRegressor", "ai.onnx.ml", 3)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


class _ZipMap(StandardNode):
    @dataclass
    class Attributes:
        classlabels_int64s: Optional[AttrInt64s]
        classlabels_strings: Optional[AttrStrings]

    class Inputs(ArrowFields):
        X: Arrow

    class Outputs(ArrowFields):
        Z: Arrow

    op_type = OpType("ZipMap", "ai.onnx.ml", 1)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def array_feature_extractor(
    X: Arrow,
    Y: Arrow,
) -> Arrow:
    r"""
    Select elements of the input tensor based on the indices passed.


        The indices are applied to the last axes of the tensor.

    Parameters
    ==========
    X
        Type T.
        Data to be selected
    Y
        Type tensor(int64).
        The indices, based on 0 as the first index of any dimension.

    Returns
    =======
    Z : Arrow
        Type T.
        Selected output data as an array

    Notes
    =====
    Signature: ``ai.onnx.ml@1::ArrayFeatureExtractor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`, `tensor(string)`
    """
    return _ArrayFeatureExtractor(
        _ArrayFeatureExtractor.Attributes(),
        _ArrayFeatureExtractor.Inputs(
            X=X,
            Y=Y,
        ),
    ).outputs.Z


def binarizer(
    X: Arrow,
    *,
    threshold: float = 0.0,
) -> Arrow:
    r"""
    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

    Parameters
    ==========
    X
        Type T.
        Data to be binarized
    threshold
        Attribute.
        Values greater than this are mapped to 1, others to 0.

    Returns
    =======
    Y : Arrow
        Type T.
        Binarized output data

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Binarizer``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Binarizer(
        _Binarizer.Attributes(
            threshold=None if threshold is None else AttrFloat32(threshold),
        ),
        _Binarizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def cast_map(
    X: Arrow,
    *,
    cast_to: str = "TO_FLOAT",
    map_form: str = "DENSE",
    max_map: int = 1,
) -> Arrow:
    r"""
    Converts a map to a tensor.

    The map key must be an int64 and the values will be ordered
        in ascending order based on this key.

    The operator supports dense packing or sparse packing.
        If using sparse packing, the key cannot exceed the max_map-1 value.

    Parameters
    ==========
    X
        Type T1.
        The input map that is to be cast to a tensor
    cast_to
        Attribute.
        A string indicating the desired element type of the output tensor, one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'.
    map_form
        Attribute.
        Indicates whether to only output as many values as are in the input (dense), or position the input based on using the key of the map as the index of the output (sparse).

        One of 'DENSE', 'SPARSE'.
    max_map
        Attribute.
        If the value of map_form is 'SPARSE,' this attribute indicates the total length of the output tensor.

    Returns
    =======
    Y : Arrow
        Type T2.
        A tensor representing the same data as the input map, ordered by their keys

    Notes
    =====
    Signature: ``ai.onnx.ml@1::CastMap``.

    Type constraints:
     - T1: `map(int64,tensor(float))`, `map(int64,tensor(string))`
     - T2: `tensor(float)`, `tensor(int64)`, `tensor(string)`
    """
    return _CastMap(
        _CastMap.Attributes(
            cast_to=None if cast_to is None else AttrString(cast_to),
            map_form=None if map_form is None else AttrString(map_form),
            max_map=None if max_map is None else AttrInt64(max_map),
        ),
        _CastMap.Inputs(
            X=X,
        ),
    ).outputs.Y


def category_mapper(
    X: Arrow,
    *,
    cats_int64s: Optional[Iterable[int]] = None,
    cats_strings: Optional[Iterable[str]] = None,
    default_int64: int = -1,
    default_string: str = "_Unused",
) -> Arrow:
    r"""
    Converts strings to integers and vice versa.


        Two sequences of equal length are used to map between integers and strings,
        with strings and integers at the same index detailing the mapping.


        Each operator converts either integers to strings or strings to integers, depending
        on which default value attribute is provided. Only one default value attribute
        should be defined.


        If the string default value is set, it will convert integers to strings.
        If the int default value is set, it will convert strings to integers.

    Parameters
    ==========
    X
        Type T1.
        Input data
    cats_int64s
        Attribute.
        The integers of the map. This sequence must be the same length as the 'cats_strings' sequence.
    cats_strings
        Attribute.
        The strings of the map. This sequence must be the same length as the 'cats_int64s' sequence
    default_int64
        Attribute.
        An integer to use when an input string value is not found in the map.

        One and only one of the 'default_*' attributes must be defined.
    default_string
        Attribute.
        A string to use when an input integer value is not found in the map.

        One and only one of the 'default_*' attributes must be defined.

    Returns
    =======
    Y : Arrow
        Type T2.
        Output data. If strings are input, the output values are integers, and vice versa.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::CategoryMapper``.

    Type constraints:
     - T1: `tensor(int64)`, `tensor(string)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _CategoryMapper(
        _CategoryMapper.Attributes(
            cats_int64s=None if cats_int64s is None else AttrInt64s(cats_int64s),
            cats_strings=None if cats_strings is None else AttrStrings(cats_strings),
            default_int64=None if default_int64 is None else AttrInt64(default_int64),
            default_string=None
            if default_string is None
            else AttrString(default_string),
        ),
        _CategoryMapper.Inputs(
            X=X,
        ),
    ).outputs.Y


def dict_vectorizer(
    X: Arrow,
    *,
    int64_vocabulary: Optional[Iterable[int]] = None,
    string_vocabulary: Optional[Iterable[str]] = None,
) -> Arrow:
    r"""
    Uses an index mapping to convert a dictionary to an array.


        Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
        the key type. The index into the vocabulary array at which the key is found is then
        used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.


        The key type of the input map must correspond to the element type of the defined vocabulary attribute.
        Therefore, the output array will be equal in length to the index mapping vector parameter.
        All keys in the input dictionary must be present in the index mapping vector.
        For each item in the input dictionary, insert its value in the output array.
        Any keys not present in the input dictionary, will be zero in the output array.


        For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
        then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.

    Parameters
    ==========
    X
        Type T1.
        A dictionary.
    int64_vocabulary
        Attribute.
        An integer vocabulary array.

        One and only one of the vocabularies must be defined.
    string_vocabulary
        Attribute.
        A string vocabulary array.

        One and only one of the vocabularies must be defined.

    Returns
    =======
    Y : Arrow
        Type T2.
        A 1-D tensor holding values from the input dictionary.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::DictVectorizer``.

    Type constraints:
     - T1: `map(int64,tensor(double))`, `map(int64,tensor(float))`, `map(int64,tensor(string))`, `map(string,tensor(double))`, `map(string,tensor(float))`, `map(string,tensor(int64))`
     - T2: `tensor(double)`, `tensor(float)`, `tensor(int64)`, `tensor(string)`
    """
    return _DictVectorizer(
        _DictVectorizer.Attributes(
            int64_vocabulary=None
            if int64_vocabulary is None
            else AttrInt64s(int64_vocabulary),
            string_vocabulary=None
            if string_vocabulary is None
            else AttrStrings(string_vocabulary),
        ),
        _DictVectorizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def feature_vectorizer(
    X: Sequence[Arrow],
    *,
    inputdimensions: Optional[Iterable[int]] = None,
) -> Arrow:
    r"""
    Concatenates input tensors into one continuous output.


        All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
        Inputs are copied to the output maintaining the order of the input arguments.


        All inputs must be integers or floats, while the output will be all floating point values.

    Parameters
    ==========
    X
        Type T1.
        An ordered collection of tensors, all with the same element type.
    inputdimensions
        Attribute.
        The size of each input in the input list

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        The output array, elements ordered as the inputs.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::FeatureVectorizer``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _FeatureVectorizer(
        _FeatureVectorizer.Attributes(
            inputdimensions=None
            if inputdimensions is None
            else AttrInt64s(inputdimensions),
        ),
        _FeatureVectorizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def imputer(
    X: Arrow,
    *,
    imputed_value_floats: Optional[Iterable[float]] = None,
    imputed_value_int64s: Optional[Iterable[int]] = None,
    replaced_value_float: float = 0.0,
    replaced_value_int64: int = 0,
) -> Arrow:
    r"""
    Replaces inputs that equal one value with another, leaving all other elements alone.


        This operator is typically used to replace missing values in situations where they have a canonical
        representation, such as -1, 0, NaN, or some extreme value.


        One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
        holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
        width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
        which one depends on whether floats or integers are being processed.


        The imputed_value attribute length can be 1 element, or it can have one element per input feature.

    In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.

    Parameters
    ==========
    X
        Type T.
        Data to be processed.
    imputed_value_floats
        Attribute.
        Value(s) to change to
    imputed_value_int64s
        Attribute.
        Value(s) to change to.
    replaced_value_float
        Attribute.
        A value that needs replacing.
    replaced_value_int64
        Attribute.
        A value that needs replacing.

    Returns
    =======
    Y : Arrow
        Type T.
        Imputed output data

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Imputer``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Imputer(
        _Imputer.Attributes(
            imputed_value_floats=None
            if imputed_value_floats is None
            else AttrFloat32s(imputed_value_floats),
            imputed_value_int64s=None
            if imputed_value_int64s is None
            else AttrInt64s(imputed_value_int64s),
            replaced_value_float=None
            if replaced_value_float is None
            else AttrFloat32(replaced_value_float),
            replaced_value_int64=None
            if replaced_value_int64 is None
            else AttrInt64(replaced_value_int64),
        ),
        _Imputer.Inputs(
            X=X,
        ),
    ).outputs.Y


def label_encoder(
    X: Arrow,
    *,
    default_float: float = -0.0,
    default_int64: int = -1,
    default_string: str = "_Unused",
    keys_floats: Optional[Iterable[float]] = None,
    keys_int64s: Optional[Iterable[int]] = None,
    keys_strings: Optional[Iterable[str]] = None,
    values_floats: Optional[Iterable[float]] = None,
    values_int64s: Optional[Iterable[int]] = None,
    values_strings: Optional[Iterable[str]] = None,
) -> Arrow:
    r"""
    Maps each element in the input tensor to another value.


        The mapping is determined by the two parallel attributes, 'keys_*' and
        'values_*' attribute. The i-th value in the specified 'keys_*' attribute
        would be mapped to the i-th value in the specified 'values_*' attribute. It
        implies that input's element type and the element type of the specified
        'keys_*' should be identical while the output type is identical to the
        specified 'values_*' attribute. If an input element can not be found in the
        specified 'keys_*' attribute, the 'default_*' that matches the specified
        'values_*' attribute may be used as its output value.


        Let's consider an example which maps a string tensor to an integer tensor.
        Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
        and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
        "Sally"] would be mapped to [-1, 5, 5, 6, 6].


        Since this operator is an one-to-one mapping, its input and output shapes
        are the same. Notice that only one of 'keys_*'/'values_*' can be set.


        For key look-up, bit-wise comparison is used so even a float NaN can be
        mapped to a value in 'values_*' attribute.



    Parameters
    ==========
    X
        Type T1.
        Input data. It can be either tensor or scalar.
    default_float
        Attribute.
        A float.
    default_int64
        Attribute.
        An integer.
    default_string
        Attribute.
        A string.
    keys_floats
        Attribute.
        A list of floats.
    keys_int64s
        Attribute.
        A list of ints.
    keys_strings
        Attribute.
        A list of strings. One and only one of 'keys_*'s should be set.
    values_floats
        Attribute.
        A list of floats.
    values_int64s
        Attribute.
        A list of ints.
    values_strings
        Attribute.
        A list of strings. One and only one of 'value_*'s should be set.

    Returns
    =======
    Y : Arrow
        Type T2.
        Output data.

    Notes
    =====
    Signature: ``ai.onnx.ml@2::LabelEncoder``.

    Type constraints:
     - T1: `tensor(float)`, `tensor(int64)`, `tensor(string)`
     - T2: `tensor(float)`, `tensor(int64)`, `tensor(string)`
    """
    return _LabelEncoder(
        _LabelEncoder.Attributes(
            default_float=None if default_float is None else AttrFloat32(default_float),
            default_int64=None if default_int64 is None else AttrInt64(default_int64),
            default_string=None
            if default_string is None
            else AttrString(default_string),
            keys_floats=None if keys_floats is None else AttrFloat32s(keys_floats),
            keys_int64s=None if keys_int64s is None else AttrInt64s(keys_int64s),
            keys_strings=None if keys_strings is None else AttrStrings(keys_strings),
            values_floats=None
            if values_floats is None
            else AttrFloat32s(values_floats),
            values_int64s=None if values_int64s is None else AttrInt64s(values_int64s),
            values_strings=None
            if values_strings is None
            else AttrStrings(values_strings),
        ),
        _LabelEncoder.Inputs(
            X=X,
        ),
    ).outputs.Y


def linear_classifier(
    X: Arrow,
    *,
    classlabels_ints: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
    coefficients: Iterable[float],
    intercepts: Optional[Iterable[float]] = None,
    multi_class: int = 0,
    post_transform: str = "NONE",
) -> _LinearClassifier.Outputs:
    r"""
    Linear classifier

    Parameters
    ==========
    X
        Type T1.
        Data to be classified.
    classlabels_ints
        Attribute.
        Class labels when using integer labels. One and only one 'classlabels' attribute must be defined.
    classlabels_strings
        Attribute.
        Class labels when using string labels. One and only one 'classlabels' attribute must be defined.
    coefficients
        Attribute.
        A collection of weights of the model(s).
    intercepts
        Attribute.
        A collection of intercepts.
    multi_class
        Attribute.
        Indicates whether to do OvR or multinomial (0=OvR is the default).
    post_transform
        Attribute.
        Indicates the transform to apply to the scores vector.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'

    Returns
    =======
    Y : Arrow
        Type T2.
        Classification outputs (one class per example).
    Z : Arrow
        Type tensor(float).
        Classification scores ([N,E] - one score for each class and example

    Notes
    =====
    Signature: ``ai.onnx.ml@1::LinearClassifier``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _LinearClassifier(
        _LinearClassifier.Attributes(
            classlabels_ints=None
            if classlabels_ints is None
            else AttrInt64s(classlabels_ints),
            classlabels_strings=None
            if classlabels_strings is None
            else AttrStrings(classlabels_strings),
            coefficients=None if coefficients is None else AttrFloat32s(coefficients),
            intercepts=None if intercepts is None else AttrFloat32s(intercepts),
            multi_class=None if multi_class is None else AttrInt64(multi_class),
            post_transform=None
            if post_transform is None
            else AttrString(post_transform),
        ),
        _LinearClassifier.Inputs(
            X=X,
        ),
    ).outputs


def linear_regressor(
    X: Arrow,
    *,
    coefficients: Optional[Iterable[float]] = None,
    intercepts: Optional[Iterable[float]] = None,
    post_transform: str = "NONE",
    targets: int = 1,
) -> Arrow:
    r"""
    Generalized linear regression evaluation.


        If targets is set to 1 (default) then univariate regression is performed.


        If targets is set to M then M sets of coefficients must be passed in as a sequence
        and M results will be output for each input n in N.


        The coefficients array is of length n, and the coefficients for each target are contiguous.
        Intercepts are optional but if provided must match the number of targets.

    Parameters
    ==========
    X
        Type T.
        Data to be regressed.
    coefficients
        Attribute.
        Weights of the model(s).
    intercepts
        Attribute.
        Weights of the intercepts, if used.
    post_transform
        Attribute.
        Indicates the transform to apply to the regression output vector.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
    targets
        Attribute.
        The total number of regression targets, 1 if not defined.

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Regression outputs (one per target, per example).

    Notes
    =====
    Signature: ``ai.onnx.ml@1::LinearRegressor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _LinearRegressor(
        _LinearRegressor.Attributes(
            coefficients=None if coefficients is None else AttrFloat32s(coefficients),
            intercepts=None if intercepts is None else AttrFloat32s(intercepts),
            post_transform=None
            if post_transform is None
            else AttrString(post_transform),
            targets=None if targets is None else AttrInt64(targets),
        ),
        _LinearRegressor.Inputs(
            X=X,
        ),
    ).outputs.Y


def normalizer(
    X: Arrow,
    *,
    norm: str = "MAX",
) -> Arrow:
    r"""
    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
        defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':





        Max: Y = X / max(X)


        L1:  Y = X / sum(X)


        L2:  Y = sqrt(X^2 / sum(X^2)}


        In all modes, if the divisor is zero, Y == X.



        For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
        of the batch is normalized independently.

    Parameters
    ==========
    X
        Type T.
        Data to be encoded, a tensor of shape [N,C] or [C]
    norm
        Attribute.
        One of 'MAX,' 'L1,' 'L2'

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Encoded output data

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Normalizer``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Normalizer(
        _Normalizer.Attributes(
            norm=None if norm is None else AttrString(norm),
        ),
        _Normalizer.Inputs(
            X=X,
        ),
    ).outputs.Y


def one_hot_encoder(
    X: Arrow,
    *,
    cats_int64s: Optional[Iterable[int]] = None,
    cats_strings: Optional[Iterable[str]] = None,
    zeros: int = 1,
) -> Arrow:
    r"""
    Replace each input element with an array of ones and zeros, where a single
        one is placed at the index of the category that was passed in. The total category count
        will determine the size of the extra dimension of the output array Y.


        For example, if we pass a tensor with a single value of 4, and a category count of 8,
        the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.


        This operator assumes every input feature is from the same set of categories.


        If the input is a tensor of float, int32, or double, the data will be cast
        to integers and the cats_int64s category list will be used for the lookups.

    Parameters
    ==========
    X
        Type T.
        Data to be encoded.
    cats_int64s
        Attribute.
        List of categories, ints.

        One and only one of the 'cats_*' attributes must be defined.
    cats_strings
        Attribute.
        List of categories, strings.

        One and only one of the 'cats_*' attributes must be defined.
    zeros
        Attribute.
        If true and category is not present, will return all zeros; if false and a category if not found, the operator will fail.

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Encoded output data, having one more dimension than X.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::OneHotEncoder``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`, `tensor(string)`
    """
    return _OneHotEncoder(
        _OneHotEncoder.Attributes(
            cats_int64s=None if cats_int64s is None else AttrInt64s(cats_int64s),
            cats_strings=None if cats_strings is None else AttrStrings(cats_strings),
            zeros=None if zeros is None else AttrInt64(zeros),
        ),
        _OneHotEncoder.Inputs(
            X=X,
        ),
    ).outputs.Y


def svmclassifier(
    X: Arrow,
    *,
    classlabels_ints: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
    coefficients: Optional[Iterable[float]] = None,
    kernel_params: Optional[Iterable[float]] = None,
    kernel_type: str = "LINEAR",
    post_transform: str = "NONE",
    prob_a: Optional[Iterable[float]] = None,
    prob_b: Optional[Iterable[float]] = None,
    rho: Optional[Iterable[float]] = None,
    support_vectors: Optional[Iterable[float]] = None,
    vectors_per_class: Optional[Iterable[int]] = None,
) -> _SVMClassifier.Outputs:
    r"""
    Support Vector Machine classifier

    Parameters
    ==========
    X
        Type T1.
        Data to be classified.
    classlabels_ints
        Attribute.
        Class labels if using integer labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    classlabels_strings
        Attribute.
        Class labels if using string labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    coefficients
        Attribute.

    kernel_params
        Attribute.
        List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.
    kernel_type
        Attribute.
        The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
    prob_a
        Attribute.
        First set of probability coefficients.
    prob_b
        Attribute.
        Second set of probability coefficients. This array must be same size as prob_a.

        If these are provided then output Z are probability estimates, otherwise they are raw scores.
    rho
        Attribute.

    support_vectors
        Attribute.

    vectors_per_class
        Attribute.


    Returns
    =======
    Y : Arrow
        Type T2.
        Classification outputs (one class per example).
    Z : Arrow
        Type tensor(float).
        Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::SVMClassifier``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _SVMClassifier(
        _SVMClassifier.Attributes(
            classlabels_ints=None
            if classlabels_ints is None
            else AttrInt64s(classlabels_ints),
            classlabels_strings=None
            if classlabels_strings is None
            else AttrStrings(classlabels_strings),
            coefficients=None if coefficients is None else AttrFloat32s(coefficients),
            kernel_params=None
            if kernel_params is None
            else AttrFloat32s(kernel_params),
            kernel_type=None if kernel_type is None else AttrString(kernel_type),
            post_transform=None
            if post_transform is None
            else AttrString(post_transform),
            prob_a=None if prob_a is None else AttrFloat32s(prob_a),
            prob_b=None if prob_b is None else AttrFloat32s(prob_b),
            rho=None if rho is None else AttrFloat32s(rho),
            support_vectors=None
            if support_vectors is None
            else AttrFloat32s(support_vectors),
            vectors_per_class=None
            if vectors_per_class is None
            else AttrInt64s(vectors_per_class),
        ),
        _SVMClassifier.Inputs(
            X=X,
        ),
    ).outputs


def svmregressor(
    X: Arrow,
    *,
    coefficients: Optional[Iterable[float]] = None,
    kernel_params: Optional[Iterable[float]] = None,
    kernel_type: str = "LINEAR",
    n_supports: int = 0,
    one_class: int = 0,
    post_transform: str = "NONE",
    rho: Optional[Iterable[float]] = None,
    support_vectors: Optional[Iterable[float]] = None,
) -> Arrow:
    r"""
    Support Vector Machine regression prediction and one-class SVM anomaly detection.

    Parameters
    ==========
    X
        Type T.
        Data to be regressed.
    coefficients
        Attribute.
        Support vector coefficients.
    kernel_params
        Attribute.
        List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.
    kernel_type
        Attribute.
        The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.
    n_supports
        Attribute.
        The number of support vectors.
    one_class
        Attribute.
        Flag indicating whether the regression is a one-class SVM or not.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'
    rho
        Attribute.

    support_vectors
        Attribute.
        Chosen support vectors

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Regression outputs (one score per target per example).

    Notes
    =====
    Signature: ``ai.onnx.ml@1::SVMRegressor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _SVMRegressor(
        _SVMRegressor.Attributes(
            coefficients=None if coefficients is None else AttrFloat32s(coefficients),
            kernel_params=None
            if kernel_params is None
            else AttrFloat32s(kernel_params),
            kernel_type=None if kernel_type is None else AttrString(kernel_type),
            n_supports=None if n_supports is None else AttrInt64(n_supports),
            one_class=None if one_class is None else AttrInt64(one_class),
            post_transform=None
            if post_transform is None
            else AttrString(post_transform),
            rho=None if rho is None else AttrFloat32s(rho),
            support_vectors=None
            if support_vectors is None
            else AttrFloat32s(support_vectors),
        ),
        _SVMRegressor.Inputs(
            X=X,
        ),
    ).outputs.Y


def scaler(
    X: Arrow,
    *,
    offset: Optional[Iterable[float]] = None,
    scale: Optional[Iterable[float]] = None,
) -> Arrow:
    r"""
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

    Parameters
    ==========
    X
        Type T.
        Data to be scaled.
    offset
        Attribute.
        First, offset by this.

        Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.
    scale
        Attribute.
        Second, multiply by this.

        Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.

        Must be same length as 'offset'

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        Scaled output data.

    Notes
    =====
    Signature: ``ai.onnx.ml@1::Scaler``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _Scaler(
        _Scaler.Attributes(
            offset=None if offset is None else AttrFloat32s(offset),
            scale=None if scale is None else AttrFloat32s(scale),
        ),
        _Scaler.Inputs(
            X=X,
        ),
    ).outputs.Y


def tree_ensemble_classifier(
    X: Arrow,
    *,
    base_values: Optional[Iterable[float]] = None,
    base_values_as_tensor: Optional[np.ndarray] = None,
    class_ids: Optional[Iterable[int]] = None,
    class_nodeids: Optional[Iterable[int]] = None,
    class_treeids: Optional[Iterable[int]] = None,
    class_weights: Optional[Iterable[float]] = None,
    class_weights_as_tensor: Optional[np.ndarray] = None,
    classlabels_int64s: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
    nodes_falsenodeids: Optional[Iterable[int]] = None,
    nodes_featureids: Optional[Iterable[int]] = None,
    nodes_hitrates: Optional[Iterable[float]] = None,
    nodes_hitrates_as_tensor: Optional[np.ndarray] = None,
    nodes_missing_value_tracks_true: Optional[Iterable[int]] = None,
    nodes_modes: Optional[Iterable[str]] = None,
    nodes_nodeids: Optional[Iterable[int]] = None,
    nodes_treeids: Optional[Iterable[int]] = None,
    nodes_truenodeids: Optional[Iterable[int]] = None,
    nodes_values: Optional[Iterable[float]] = None,
    nodes_values_as_tensor: Optional[np.ndarray] = None,
    post_transform: str = "NONE",
) -> _TreeEnsembleClassifier.Outputs:
    r"""
    Tree Ensemble classifier. Returns the top class for each of N inputs.


        The attributes named 'nodes_X' form a sequence of tuples, associated by
        index into the sequences, which must all be of equal length. These tuples
        define the nodes.


        Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
        A leaf may have multiple votes, where each vote is weighted by
        the associated class_weights index.


        One and only one of classlabels_strings or classlabels_int64s
        will be defined. The class_ids are indices into this list.
        All fields ending with `_as_tensor` can be used instead of the
        same parameter without the suffix if the element type is double and not float.

    Parameters
    ==========
    X
        Type T1.
        Input of shape [N,F]
    base_values
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    base_values_as_tensor
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    class_ids
        Attribute.
        The index of the class list that each weight is for.
    class_nodeids
        Attribute.
        node id that this weight is for.
    class_treeids
        Attribute.
        The id of the tree that this node is in.
    class_weights
        Attribute.
        The weight for the class in class_id.
    class_weights_as_tensor
        Attribute.
        The weight for the class in class_id.
    classlabels_int64s
        Attribute.
        Class labels if using integer labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    classlabels_strings
        Attribute.
        Class labels if using string labels.

        One and only one of the 'classlabels_*' attributes must be defined.
    nodes_falsenodeids
        Attribute.
        Child node if expression is false.
    nodes_featureids
        Attribute.
        Feature id for each node.
    nodes_hitrates
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_hitrates_as_tensor
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_missing_value_tracks_true
        Attribute.
        For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the 'true' or 'false' branch based on the value in this array.

        This attribute may be left undefined, and the defalt value is false (0) for all nodes.
    nodes_modes
        Attribute.
        The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.

        One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
    nodes_nodeids
        Attribute.
        Node id for each node. Ids may restart at zero for each tree, but it not required to.
    nodes_treeids
        Attribute.
        Tree id for each node.
    nodes_truenodeids
        Attribute.
        Child node if expression is true.
    nodes_values
        Attribute.
        Thresholds to do the splitting on for each node.
    nodes_values_as_tensor
        Attribute.
        Thresholds to do the splitting on for each node.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

         One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'

    Returns
    =======
    Y : Arrow
        Type T2.
        N, Top class for each point
    Z : Arrow
        Type tensor(float).
        The class score for each class, for each point, a tensor of shape [N,E].

    Notes
    =====
    Signature: ``ai.onnx.ml@3::TreeEnsembleClassifier``.

    Type constraints:
     - T1: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
     - T2: `tensor(int64)`, `tensor(string)`
    """
    return _TreeEnsembleClassifier(
        _TreeEnsembleClassifier.Attributes(
            base_values=None if base_values is None else AttrFloat32s(base_values),
            base_values_as_tensor=None
            if base_values_as_tensor is None
            else AttrTensor(base_values_as_tensor),
            class_ids=None if class_ids is None else AttrInt64s(class_ids),
            class_nodeids=None if class_nodeids is None else AttrInt64s(class_nodeids),
            class_treeids=None if class_treeids is None else AttrInt64s(class_treeids),
            class_weights=None
            if class_weights is None
            else AttrFloat32s(class_weights),
            class_weights_as_tensor=None
            if class_weights_as_tensor is None
            else AttrTensor(class_weights_as_tensor),
            classlabels_int64s=None
            if classlabels_int64s is None
            else AttrInt64s(classlabels_int64s),
            classlabels_strings=None
            if classlabels_strings is None
            else AttrStrings(classlabels_strings),
            nodes_falsenodeids=None
            if nodes_falsenodeids is None
            else AttrInt64s(nodes_falsenodeids),
            nodes_featureids=None
            if nodes_featureids is None
            else AttrInt64s(nodes_featureids),
            nodes_hitrates=None
            if nodes_hitrates is None
            else AttrFloat32s(nodes_hitrates),
            nodes_hitrates_as_tensor=None
            if nodes_hitrates_as_tensor is None
            else AttrTensor(nodes_hitrates_as_tensor),
            nodes_missing_value_tracks_true=None
            if nodes_missing_value_tracks_true is None
            else AttrInt64s(nodes_missing_value_tracks_true),
            nodes_modes=None if nodes_modes is None else AttrStrings(nodes_modes),
            nodes_nodeids=None if nodes_nodeids is None else AttrInt64s(nodes_nodeids),
            nodes_treeids=None if nodes_treeids is None else AttrInt64s(nodes_treeids),
            nodes_truenodeids=None
            if nodes_truenodeids is None
            else AttrInt64s(nodes_truenodeids),
            nodes_values=None if nodes_values is None else AttrFloat32s(nodes_values),
            nodes_values_as_tensor=None
            if nodes_values_as_tensor is None
            else AttrTensor(nodes_values_as_tensor),
            post_transform=None
            if post_transform is None
            else AttrString(post_transform),
        ),
        _TreeEnsembleClassifier.Inputs(
            X=X,
        ),
    ).outputs


def tree_ensemble_regressor(
    X: Arrow,
    *,
    aggregate_function: str = "SUM",
    base_values: Optional[Iterable[float]] = None,
    base_values_as_tensor: Optional[np.ndarray] = None,
    n_targets: Optional[int] = None,
    nodes_falsenodeids: Optional[Iterable[int]] = None,
    nodes_featureids: Optional[Iterable[int]] = None,
    nodes_hitrates: Optional[Iterable[float]] = None,
    nodes_hitrates_as_tensor: Optional[np.ndarray] = None,
    nodes_missing_value_tracks_true: Optional[Iterable[int]] = None,
    nodes_modes: Optional[Iterable[str]] = None,
    nodes_nodeids: Optional[Iterable[int]] = None,
    nodes_treeids: Optional[Iterable[int]] = None,
    nodes_truenodeids: Optional[Iterable[int]] = None,
    nodes_values: Optional[Iterable[float]] = None,
    nodes_values_as_tensor: Optional[np.ndarray] = None,
    post_transform: str = "NONE",
    target_ids: Optional[Iterable[int]] = None,
    target_nodeids: Optional[Iterable[int]] = None,
    target_treeids: Optional[Iterable[int]] = None,
    target_weights: Optional[Iterable[float]] = None,
    target_weights_as_tensor: Optional[np.ndarray] = None,
) -> Arrow:
    r"""
    Tree Ensemble regressor.  Returns the regressed values for each input in N.


        All args with nodes_ are fields of a tuple of tree nodes, and
        it is assumed they are the same length, and an index i will decode the
        tuple across these inputs.  Each node id can appear only once
        for each tree id.


        All fields prefixed with target_ are tuples of votes at the leaves.


        A leaf may have multiple votes, where each vote is weighted by
        the associated target_weights index.


        All fields ending with `_as_tensor` can be used instead of the
        same parameter without the suffix if the element type is double and not float.
        All trees must have their node ids start at 0 and increment by 1.


        Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF

    Parameters
    ==========
    X
        Type T.
        Input of shape [N,F]
    aggregate_function
        Attribute.
        Defines how to aggregate leaf values within a target.

        One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'
    base_values
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    base_values_as_tensor
        Attribute.
        Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
    n_targets
        Attribute.
        The total number of targets.
    nodes_falsenodeids
        Attribute.
        Child node if expression is false
    nodes_featureids
        Attribute.
        Feature id for each node.
    nodes_hitrates
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_hitrates_as_tensor
        Attribute.
        Popularity of each node, used for performance and may be omitted.
    nodes_missing_value_tracks_true
        Attribute.
        For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) or 'false' (if the attribute value is 0) branch based on the value in this array.

        This attribute may be left undefined and the defalt value is false (0) for all nodes.
    nodes_modes
        Attribute.
        The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.

        One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
    nodes_nodeids
        Attribute.
        Node id for each node. Node ids must restart at zero for each tree and increase sequentially.
    nodes_treeids
        Attribute.
        Tree id for each node.
    nodes_truenodeids
        Attribute.
        Child node if expression is true
    nodes_values
        Attribute.
        Thresholds to do the splitting on for each node.
    nodes_values_as_tensor
        Attribute.
        Thresholds to do the splitting on for each node.
    post_transform
        Attribute.
        Indicates the transform to apply to the score.

        One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
    target_ids
        Attribute.
        The index of the target that each weight is for
    target_nodeids
        Attribute.
        The node id of each weight
    target_treeids
        Attribute.
        The id of the tree that each node is in.
    target_weights
        Attribute.
        The weight for each target
    target_weights_as_tensor
        Attribute.
        The weight for each target

    Returns
    =======
    Y : Arrow
        Type tensor(float).
        N classes

    Notes
    =====
    Signature: ``ai.onnx.ml@3::TreeEnsembleRegressor``.

    Type constraints:
     - T: `tensor(double)`, `tensor(float)`, `tensor(int32)`, `tensor(int64)`
    """
    return _TreeEnsembleRegressor(
        _TreeEnsembleRegressor.Attributes(
            aggregate_function=None
            if aggregate_function is None
            else AttrString(aggregate_function),
            base_values=None if base_values is None else AttrFloat32s(base_values),
            base_values_as_tensor=None
            if base_values_as_tensor is None
            else AttrTensor(base_values_as_tensor),
            n_targets=None if n_targets is None else AttrInt64(n_targets),
            nodes_falsenodeids=None
            if nodes_falsenodeids is None
            else AttrInt64s(nodes_falsenodeids),
            nodes_featureids=None
            if nodes_featureids is None
            else AttrInt64s(nodes_featureids),
            nodes_hitrates=None
            if nodes_hitrates is None
            else AttrFloat32s(nodes_hitrates),
            nodes_hitrates_as_tensor=None
            if nodes_hitrates_as_tensor is None
            else AttrTensor(nodes_hitrates_as_tensor),
            nodes_missing_value_tracks_true=None
            if nodes_missing_value_tracks_true is None
            else AttrInt64s(nodes_missing_value_tracks_true),
            nodes_modes=None if nodes_modes is None else AttrStrings(nodes_modes),
            nodes_nodeids=None if nodes_nodeids is None else AttrInt64s(nodes_nodeids),
            nodes_treeids=None if nodes_treeids is None else AttrInt64s(nodes_treeids),
            nodes_truenodeids=None
            if nodes_truenodeids is None
            else AttrInt64s(nodes_truenodeids),
            nodes_values=None if nodes_values is None else AttrFloat32s(nodes_values),
            nodes_values_as_tensor=None
            if nodes_values_as_tensor is None
            else AttrTensor(nodes_values_as_tensor),
            post_transform=None
            if post_transform is None
            else AttrString(post_transform),
            target_ids=None if target_ids is None else AttrInt64s(target_ids),
            target_nodeids=None
            if target_nodeids is None
            else AttrInt64s(target_nodeids),
            target_treeids=None
            if target_treeids is None
            else AttrInt64s(target_treeids),
            target_weights=None
            if target_weights is None
            else AttrFloat32s(target_weights),
            target_weights_as_tensor=None
            if target_weights_as_tensor is None
            else AttrTensor(target_weights_as_tensor),
        ),
        _TreeEnsembleRegressor.Inputs(
            X=X,
        ),
    ).outputs.Y


def zip_map(
    X: Arrow,
    *,
    classlabels_int64s: Optional[Iterable[int]] = None,
    classlabels_strings: Optional[Iterable[str]] = None,
) -> Arrow:
    r"""
    Creates a map from the input and the attributes.


        The values are provided by the input tensor, while the keys are specified by the attributes.
        Must provide keys in either classlabels_strings or classlabels_int64s (but not both).


        The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.



    Parameters
    ==========
    X
        Type tensor(float).
        The input values
    classlabels_int64s
        Attribute.
        The keys when using int keys.

        One and only one of the 'classlabels_*' attributes must be defined.
    classlabels_strings
        Attribute.
        The keys when using string keys.

        One and only one of the 'classlabels_*' attributes must be defined.

    Returns
    =======
    Z : Arrow
        Type T.
        The output map

    Notes
    =====
    Signature: ``ai.onnx.ml@1::ZipMap``.

    Type constraints:
     - T: `seq(map(int64,tensor(float)))`, `seq(map(string,tensor(float)))`
    """
    return _ZipMap(
        _ZipMap.Attributes(
            classlabels_int64s=None
            if classlabels_int64s is None
            else AttrInt64s(classlabels_int64s),
            classlabels_strings=None
            if classlabels_strings is None
            else AttrStrings(classlabels_strings),
        ),
        _ZipMap.Inputs(
            X=X,
        ),
    ).outputs.Z


_OPERATORS = {
    "ArrayFeatureExtractor": _ArrayFeatureExtractor,
    "Binarizer": _Binarizer,
    "CastMap": _CastMap,
    "CategoryMapper": _CategoryMapper,
    "DictVectorizer": _DictVectorizer,
    "FeatureVectorizer": _FeatureVectorizer,
    "Imputer": _Imputer,
    "LabelEncoder": _LabelEncoder,
    "LinearClassifier": _LinearClassifier,
    "LinearRegressor": _LinearRegressor,
    "Normalizer": _Normalizer,
    "OneHotEncoder": _OneHotEncoder,
    "SVMClassifier": _SVMClassifier,
    "SVMRegressor": _SVMRegressor,
    "Scaler": _Scaler,
    "TreeEnsembleClassifier": _TreeEnsembleClassifier,
    "TreeEnsembleRegressor": _TreeEnsembleRegressor,
    "ZipMap": _ZipMap,
}

_CONSTRUCTORS = {
    "ArrayFeatureExtractor": array_feature_extractor,
    "Binarizer": binarizer,
    "CastMap": cast_map,
    "CategoryMapper": category_mapper,
    "DictVectorizer": dict_vectorizer,
    "FeatureVectorizer": feature_vectorizer,
    "Imputer": imputer,
    "LabelEncoder": label_encoder,
    "LinearClassifier": linear_classifier,
    "LinearRegressor": linear_regressor,
    "Normalizer": normalizer,
    "OneHotEncoder": one_hot_encoder,
    "SVMClassifier": svmclassifier,
    "SVMRegressor": svmregressor,
    "Scaler": scaler,
    "TreeEnsembleClassifier": tree_ensemble_classifier,
    "TreeEnsembleRegressor": tree_ensemble_regressor,
    "ZipMap": zip_map,
}

__all__ = [fun.__name__ for fun in _CONSTRUCTORS.values()]
