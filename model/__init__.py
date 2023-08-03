from ray.rllib.models.catalog import ModelCatalog

from model.gnn_rllib import GNNPolicy

from model.gnn_rllib_scout import GNNScoutPolicy
from model.graph_transformer_rllib import GraphTransformerPolicy
from model.hybrid_rllib import HybridPolicy
from model.hybrid_rllib_scout import HybridScoutPolicy
from model.fc_rllib import FCPolicy
from model.fc_rllib_scout import FCScoutPolicy
from model.two_gnn_policy_rllib_scout import TwoGNNDirectScoutPolicy
from model.gnn_policy_rllib_scout import GNNDirectScoutPolicy
from model.global_hybrid_gnn_rllib import GlobalHybridGNNPolicy
from model.correlated_gnn_rllib_scout import CorrelatedGNNScoutPolicy

# register our model (put in an __init__ file later)
# https://docs.ray.io/en/latest/rllib-models.html#customizing-preprocessors-and-models
ModelCatalog.register_custom_model("gnn_policy", GNNPolicy)
ModelCatalog.register_custom_model("gnn_scout_policy", GNNScoutPolicy)
ModelCatalog.register_custom_model("graph_transformer_policy", GraphTransformerPolicy)
ModelCatalog.register_custom_model("hybrid_policy", HybridPolicy)
ModelCatalog.register_custom_model("hybrid_scout_policy", HybridScoutPolicy)
ModelCatalog.register_custom_model("gt_policy", HybridPolicy)
ModelCatalog.register_custom_model("gt_scout_policy", HybridScoutPolicy)
ModelCatalog.register_custom_model("fc_policy", FCPolicy)
ModelCatalog.register_custom_model("fc_scout_policy", FCScoutPolicy)
ModelCatalog.register_custom_model("two_gnn_direct_scout_policy", TwoGNNDirectScoutPolicy)
ModelCatalog.register_custom_model("gnn_direct_scout_policy", GNNDirectScoutPolicy)
ModelCatalog.register_custom_model("global_hybrid_policy", GlobalHybridGNNPolicy)
ModelCatalog.register_custom_model("correlated_gnn_scout_policy", CorrelatedGNNScoutPolicy)

"""
not under development: "attention learn to route"-based model
"""
# from model.altr_rllib import AltrPolicy
# ModelCatalog.register_custom_model("altr_policy", AltrPolicy)
