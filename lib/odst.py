import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nn_utils import sparsemax, sparsemoid, ModuleWithInit, entmax15, entmoid15
from .utils import check_numpy
from warnings import warn


MIN_LOGITS = -20


class ODST(ModuleWithInit):
    def __init__(self, input_dim, num_trees, depth=6, tree_dim=1, flatten_output=True,
                 choice_function=entmax15, bin_function=entmoid15,
                 initialize_response_=nn.init.normal_,
                 initialize_selection_logits_=nn.init.uniform_,
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0,
                 colsample_bytree=1., **kwargs
                 ):
        """
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param input_dim: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights

        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        :param threshold_init_cutoff: threshold log-temperatures initializer, \in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)

        :param colsample_bytree: the same argument as in xgboost package.
            If less than 1, for each tree, it will only choose a fraction of features to train. For instance,
            if colsample_bytree = 0.9, each tree will only selects among 90% of the features.
        """
        super().__init__()
        self.input_dim, self.depth, self.num_trees, self.tree_dim, self.flatten_output = \
            input_dim, depth, num_trees, tree_dim, flatten_output
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff
        self.colsample_bytree = colsample_bytree

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True)
        initialize_response_(self.response)

        self.num_sample_feats = input_dim
        if self.colsample_bytree < 1.:
            self.num_sample_feats = int(np.ceil(input_dim * self.colsample_bytree))

        # Do the subsampling
        if self.num_sample_feats < input_dim:
            self.colsample = nn.Parameter(
                torch.zeros([input_dim, num_trees, 1]), requires_grad=False
            )
            for nt in range(num_trees):
                rand_idx = torch.randperm(input_dim)[:self.num_sample_feats]
                self.colsample[rand_idx, nt, 0] = 1.

        # Only when num_sample_feats > 1, we initialize this logit
        if self.num_sample_feats > 1 or self.colsample_bytree == 1.:
            self.feature_selection_logits = nn.Parameter(
                torch.zeros([input_dim, num_trees, depth]), requires_grad=True
            )
            initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        # new input shape: [batch_size, input_dim]

        feature_values = self.get_feature_selection_values(input)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]
        
        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                 "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                 "You can do so manually before training. Use with torch.no_grad() for memory efficiency.")
        with torch.no_grad():
            feature_values = self.get_feature_selection_values(input)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,
                                                 size=[self.num_trees, self.depth])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())),
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def get_feature_selection_values(self, input):
        ''' Select which features to split '''
        feature_selectors = self.get_feature_selectors()
        # ^--[input_dim, num_trees, depth]

        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
        return feature_values

    def get_feature_selectors(self):
        if self.colsample_bytree < 1. and self.num_sample_feats == 1:
            return self.colsample.data

        fsl = self.feature_selection_logits
        if self.colsample_bytree < 1.:
            fsl = self.colsample * fsl \
                  + (1. - self.colsample) * MIN_LOGITS
        feature_selectors = self.choice_function(fsl, dim=0)
        return feature_selectors

    def __repr__(self):
        return "{}(input_dim={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__, self.input_dim,
            self.num_trees, self.depth, self.tree_dim, self.flatten_output
        )


class GAM_ODST(ODST):
    def __init__(self, input_dim, *args,
                 depth=3,
                 colsample_bytree=1.,
                 initialize_selection_logits_=nn.init.uniform_,
                 selectors_detach=False, fs_normalize=True,
                 ga2m=0, **kwargs):
        '''
        Change an ODST tree that depends on only 1 or 2 features.

        selectors_detach: if True, the selector will be detached before passing into the next layer.
            This will save GPU memory in the large dataset (e.g. Epsilon).
        fs_normalize: if True, we normalize the feature selectors be summed to 1. But actually 
            False or True do not make too much difference.
        ga2m: if set to 1, use GA2M, else use GAM.
        '''
        if ga2m:
            # If the colsample_bytree too small or depth < 2 that there are not at least 2 features 
            # modeled in the tree, just downgrade to GAM.
            if depth < 2 \
                    or (colsample_bytree < 1. and int(np.ceil(input_dim * colsample_bytree)) < 2):
                print('Use GAM instead since colsample_by_tree is too small or depth=1 that GA2M is not allowed.')
                ga2m = 0

        super().__init__(
            input_dim,
            *args,
            depth=depth,
            colsample_bytree=colsample_bytree,
            initialize_selection_logits_=initialize_selection_logits_,
            **kwargs)
        self.selectors_detach = selectors_detach
        self.fs_normalize = fs_normalize
        self.ga2m = ga2m

        # Remove the feature_selection logits defined in the ODST and instead re-initialize to only at most
        # 1 or 2 depths,
        del self.feature_selection_logits
        the_depth = 1 if not self.ga2m else 2
        self.feature_selection_logits = nn.Parameter(
            torch.zeros([self.input_dim, self.num_trees, the_depth]), requires_grad=True
        )
        initialize_selection_logits_(self.feature_selection_logits)

    def forward(self, input, return_feature_selectors=True, prev_feature_selectors=None):
        self.prev_feature_selectors = prev_feature_selectors

        response = super().forward(input)

        fs, self.feature_selectors = self.feature_selectors, None
        if return_feature_selectors:
            return response, fs

        return response

    def initialize(self, input, return_feature_selectors=True,
                   prev_feature_selectors=None, eps=1e-6):
        self.prev_feature_selectors = prev_feature_selectors
        response = super().initialize(input, eps=eps)
        self.feature_selectors = None

    def get_feature_selection_values(self, input, return_fss=False):
        feature_selectors = self.get_feature_selectors()
        # ^--[input_dim, num_trees, depth=1]

        # A hack to pass this value outside of this function
        self.feature_selectors = feature_selectors
        if self.selectors_detach: # To save memory
            self.feature_selectors = self.feature_selectors.detach()

        # It needs to multiply by the tree_dim
        if self.tree_dim > 1:
            shape = self.feature_selectors.shape
            self.feature_selectors = self.feature_selectors.unsqueeze(-2).expand(
                -1, -1, self.tree_dim, -1
            ).reshape(shape[0], -1, shape[-1])
            # ^--[input_dim, num_trees * tree_dim, depth]

        if input.shape[1] > self.input_dim:  # The rest are previous layers
            # Check incoming data
            pfs, self.prev_feature_selectors = self.prev_feature_selectors, None
            assert pfs.shape[:2] == (self.input_dim, input.shape[1] - self.input_dim), \
                'Previous selectors does not have the same shape as the input: %s != %s' \
                % (pfs.shape[:2], (self.input_dim, input.shape[1] - self.input_dim))
            fw = self.cal_prev_feat_weights(feature_selectors, pfs)

            feature_selectors = torch.cat([feature_selectors, fw], dim=0)
            # ^--[input_features, num_trees, depth=1]

        # post_process it
        feature_selectors = self.post_process(feature_selectors)

        fv = torch.einsum('bi,ind->bnd', input, feature_selectors)
        # ^--[batch_size, num_trees, depth=1,2]
        if not self.ga2m:
            fv = fv.expand(-1, -1, self.depth)
        else:
            if self.depth > 2:
                fv = fv.repeat(1, 1, int(np.ceil(self.depth / 2)))[..., :self.depth]

        if return_fss:
            return fv, feature_selectors
        return fv

    def cal_prev_feat_weights(self, myfs, pfs):
        # Do a row-wise inner product between prev selectors and cur ones
        if not self.ga2m:
            fw = torch.einsum('icd,ipd->pcd', myfs, pfs)
        else:
            g1 = torch.einsum("dp,dc->pc", pfs[:, :, 0], myfs[:, :, 0])
            g2 = torch.einsum("dp,dc->pc", pfs[:, :, 1], myfs[:, :, 1])
            g3 = torch.einsum("dp,dc->pc", pfs[:, :, 1], myfs[:, :, 0])
            g4 = torch.einsum("dp,dc->pc", pfs[:, :, 0], myfs[:, :, 1])

            fw = g1 * g2 + g3 * g4
            fw = fw.clamp_(max=1.).unsqueeze_(-1).repeat(1, 1, 2)
        return fw

    def post_process(self, feature_selectors):
        result = feature_selectors
        if self.fs_normalize:
            result = (feature_selectors / feature_selectors.sum(dim=0, keepdims=True))
        return result

    def get_num_trees_assigned_to_each_feature(self):
        with torch.no_grad():
            fs = self.get_feature_selectors()
            # ^-- [input_dim, num_trees, 1]
            return (fs > 0).sum(dim=[1, 2])


class GAMAttODST(GAM_ODST):
    def __init__(self, *args,
                 prev_input_dim=0,
                 dim_att=-1,
                 initialize_selection_logits_=nn.init.uniform_,
                 **kwargs):
        kwargs['fs_normalize'] = False # No need normalization
        super().__init__(*args,
                         initialize_selection_logits_=initialize_selection_logits_,
                         **kwargs)

        self.prev_input_dim = prev_input_dim
        self.dim_att = dim_att
        if prev_input_dim > 0:
            # Save parameter
            self.att_key = nn.Parameter(
                torch.zeros([prev_input_dim, dim_att]), requires_grad=True
            )
            self.att_query = nn.Parameter(
                torch.zeros([dim_att, self.num_trees]), requires_grad=True
            )
            initialize_selection_logits_(self.att_key)
            initialize_selection_logits_(self.att_query)

    def cal_prev_feat_weights(self, feature_selectors, pfs):
        assert self.prev_input_dim > 0

        fw = super().cal_prev_feat_weights(feature_selectors, pfs)
        # ^--[prev_in_feats, num_trees, depth=1,2]

        pfa = torch.einsum('pa,at->pt', self.att_key, self.att_query)
        new_fw = entmax15(fw.add(1e-20).log().add(pfa.unsqueeze_(-1)), dim=0)
        # new_fw = entmax15((MIN_LOGITS * (1. - fw) + fw * 0.).add(pfa.unsqueeze_(-1)), dim=0)
        fw = fw * new_fw
        return fw



class GAMAtt2ODST(GAM_ODST):
    '''
    For compatability purpose: do not worry about it.
    '''
    def __init__(self, *args,
                 prev_input_dim=0,
                 dim_att=-1,
                 initialize_selection_logits_=nn.init.uniform_,
                 **kwargs):
        kwargs['fs_normalize'] = False # No need normalization
        super().__init__(*args,
                         initialize_selection_logits_=initialize_selection_logits_,
                         **kwargs)

        self.prev_input_dim = prev_input_dim
        self.dim_att = dim_att
        if prev_input_dim > 0:
            # Save parameter
            the_depth = 1 if not self.ga2m else 2
            self.att_key = nn.Parameter(
                torch.zeros([prev_input_dim, dim_att, the_depth]), requires_grad=True
            )
            self.att_query = nn.Parameter(
                torch.zeros([dim_att, self.num_trees, the_depth]), requires_grad=True
            )
            initialize_selection_logits_(self.att_key)
            initialize_selection_logits_(self.att_query)

    def cal_prev_feat_weights(self, feature_selectors, pfs):
        assert self.prev_input_dim > 0

        fw = super().cal_prev_feat_weights(feature_selectors, pfs)
        # ^--[prev_in_feats, num_trees, depth=1,2]

        pfa = torch.einsum('pad,atd->ptd', self.att_key, self.att_query)
        new_fw = entmax15(fw.add(1e-20).log().add(pfa), dim=0)
        # new_fw = entmax15((MIN_LOGITS * (1. - fw) + fw * 0.).add(pfa.unsqueeze_(-1)), dim=0)
        fw = fw * new_fw
        return fw


class GAMAtt3ODST(GAM_ODST):
    '''
    Testing with another idea that attention depends on both input features 
        and the outputs from previous layers. Does not have too much difference
        to GAMAttODST. So no need to use it.
    '''
    def __init__(self, *args,
                 prev_input_dim=0,
                 dim_att=-1,
                 initialize_selection_logits_=nn.init.uniform_,
                 **kwargs):
        super().__init__(*args,
                         initialize_selection_logits_=initialize_selection_logits_,
                         **kwargs)
        self.prev_input_dim = prev_input_dim
        self.dim_att = dim_att

        if prev_input_dim > 0:
            the_depth = 1 if not self.ga2m else 2
            self.att_key = nn.Parameter(
                torch.zeros([self.input_dim + prev_input_dim, dim_att, the_depth]),
                requires_grad=True,
            )
            self.att_query = nn.Parameter(
                torch.zeros([dim_att, self.num_trees, the_depth]), requires_grad=True
            )
            initialize_selection_logits_(self.att_key)
            initialize_selection_logits_(self.att_query)

    def post_process(self, feature_selectors):
        fs = feature_selectors
        if self.prev_input_dim > 0:
            pfa = torch.einsum('pad,atd->ptd', self.att_key, self.att_query)
            new_fs = entmax15(fs.add(1e-20).log().add(pfa), dim=0)
            fs = new_fs * fs
        return fs

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict)
