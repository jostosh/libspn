# Import public interface of the library

# Graph
from libspn.graph.scope import Scope
from libspn.graph.node import Input
from libspn.graph.node import Node
from libspn.graph.node import OpNode
from libspn.graph.node import VarNode
from libspn.graph.node import ParamNode
from libspn.graph.op.concat import Concat
from libspn.graph.op.sum import Sum
from libspn.graph.op.parsums import ParSums
from libspn.graph.op.sumslayer import SumsLayer
from libspn.graph.op.product import Product
from libspn.graph.op.permproducts import PermProducts
from libspn.graph.op.productslayer import ProductsLayer
from libspn.graph.convsum import ConvSum
from libspn.graph.localsum import LocalSum
from libspn.graph.op.block_sum import BlockSum
from libspn.graph.op.block_permute_product import BlockPermuteProduct
from libspn.graph.op.block_reduce_product import BlockReduceProduct
from libspn.graph.op.block_random_decompositions import BlockRandomDecompositions
from libspn.graph.op.block_merge_decomps import BlockMergeDecomps
from libspn.graph.op.block_root_sum import BlockRootSum
from libspn.graph.convprod2d import ConvProd2D, _ConvProdNaive
from libspn.graph.convproddepthwise import ConvProdDepthWise
from libspn.graph.spatialpermproducts import SpatialPermProducts
from libspn.graph.stridedslice import StridedSlice2D
from libspn.graph.weights import Weights
from libspn.graph.weights import assign_weights
from libspn.graph.weights import initialize_weights
from libspn.graph.serialization import serialize_graph
from libspn.graph.serialization import deserialize_graph
from libspn.graph.saver import Saver, JSONSaver
from libspn.graph.loader import Loader, JSONLoader
from libspn.graph.algorithms import compute_graph_up
from libspn.graph.algorithms import compute_graph_up_down
from libspn.graph.algorithms import traverse_graph
from libspn.graph.leaf.normal import NormalLeaf
from libspn.graph.leaf.cauchy import CauchyLeaf
from libspn.graph.leaf.student_t import StudentTLeaf
from libspn.graph.leaf.laplace import LaplaceLeaf
from libspn.graph.leaf.indicator import IndicatorLeaf
from libspn.graph.leaf.raw import RawLeaf
from libspn.graph.leaf.multivariate_cauchy_diag import MultivariateCauchyDiagLeaf
from libspn.graph.leaf.multivariate_normal_diag import MultivariateNormalDiagLeaf

# Generators
from libspn.generation.dense import DenseSPNGenerator
from libspn.generation.conversion import convert_to_layer_nodes
from libspn.generation.weights import WeightsGenerator
from libspn.generation.weights import generate_weights

# Inference and learning
from libspn.inference.type import InferenceType
from libspn.inference.value import Value
from libspn.inference.value import LogValue
from libspn.inference.mpe_path import MPEPath
from libspn.inference.mpe_state import MPEState
from libspn.inference.gradient import Gradient
from libspn.learning.em import EMLearning
from libspn.learning.gd import GDLearning
from libspn.learning.type import LearningTaskType
from libspn.learning.type import LearningMethodType
from libspn.learning.ebw import ExtendedBaumWelch

# Data
from libspn.data.dataset import Dataset
from libspn.data.file import FileDataset
from libspn.data.csv import CSVFileDataset
from libspn.data.generated import GaussianMixtureDataset
from libspn.data.generated import IntGridDataset
from libspn.data.image import ImageFormat
from libspn.data.image import ImageShape
from libspn.data.image import ImageDatasetBase
from libspn.data.image import ImageDataset
from libspn.data.mnist import MNISTDataset
from libspn.data.cifar import CIFAR10Dataset
from libspn.data.writer import DataWriter
from libspn.data.writer import CSVDataWriter
from libspn.data.writer import ImageDataWriter

# Models
from libspn.models.model import Model
from libspn.models.discrete_dense import DiscreteDenseModel
from libspn.models.test import Poon11NaiveMixtureModel

# Session
from libspn.session import session

# Visualization
from libspn.visual.plot import plot_2d
from libspn.visual.image import show_image
from libspn.visual.tf_graph import display_tf_graph
from libspn.visual.spn_graph import display_spn_graph

# Logging
from libspn.log import config_logger
from libspn.log import get_logger
from libspn.log import WARNING
from libspn.log import INFO
from libspn.log import DEBUG1
from libspn.log import DEBUG2

# Utils and config
from libspn import conf
from libspn import utils

# App
from libspn.app import App

# Exceptions
from libspn.exceptions import StructureError

from libspn.utils.serialization import register_serializable

# Initilaizers
from libspn.utils.initializers import Equidistant

# Graphkeys
from libspn.utils.graphkeys import SPNGraphKeys


# All
__all__ = [
    # Graph
    'Scope', 'Input', 'Node', 'ParamNode', 'OpNode', 'VarNode',
    'Concat', 'IndicatorLeaf', 'RawLeaf',
    'Sum', 'ParSums', 'SumsLayer',
    'BlockSum', 'BlockPermuteProduct', 'BlockRandomDecompositions', 'BlockMergeDecomps', 'BlockRootSum',
    'Product', 'PermProducts', 'ProductsLayer',
    'Weights', 'assign_weights', 'initialize_weights',
    'serialize_graph', 'deserialize_graph',
    'Saver', 'Loader', 'JSONSaver', 'JSONLoader',
    'compute_graph_up', 'compute_graph_up_down',
    'traverse_graph',
    'StudentTLeaf', 'NormalLeaf', 'CauchyLeaf', 'LaplaceLeaf',
    'MultivariateCauchyDiagLeaf',
    # Generators
    'DenseSPNGenerator',
    'WeightsGenerator', 'generate_weights',
    # Inference and learning
    'InferenceType', 'Value', 'LogValue', 'MPEPath', 'Gradient',
    'MPEState', 'EMLearning', 'GDLearning', 'LearningTaskType',
    'LearningMethodType',
    # Data
    'Dataset', 'FileDataset', 'CSVFileDataset', 'GaussianMixtureDataset',
    'IntGridDataset', 'ImageFormat', 'ImageShape', 'ImageDatasetBase',
    'ImageDataset', 'MNISTDataset', 'CIFAR10Dataset',
    'DataWriter', 'CSVDataWriter', 'ImageDataWriter',
    # Models
    'Model', 'DiscreteDenseModel', 'Poon11NaiveMixtureModel',
    # Session
    'session',
    # Visualization
    'plot_2d', 'show_image', 'display_tf_graph', 'display_spn_graph',
    # Logging
    'config_logger', 'get_logger', 'WARNING', 'INFO', 'DEBUG1', 'DEBUG2',
    # Custom ops, utils and config
    'conf', 'utils', 'App',
    # Exceptions
    'StructureError',
    # Initializers
    'Equidistant',
    # Graphkeys
    'SPNGraphKeys']

# Configure the logger to show INFO and WARNING by default
config_logger(level=INFO)
