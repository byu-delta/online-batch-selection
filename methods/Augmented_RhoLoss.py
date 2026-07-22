import numpy as np
import torch
import torch.nn.functional as F

from methods.method_utils.build_teacher_model import build_teacher_model
from methods.method_utils.loss import *
from methods.method_utils.optimizer import *
from methods.SelectionMethod import MinibatchInfo
from methods.SelectionMethod import SelectionMethod


class Augmented_RhoLoss(SelectionMethod):
    """A class for implementing the RhoLoss selection method, which selects samples based on reducible loss.

    This class inherits from `SelectionMethod` and uses an irreducible loss model (ILmodel) and a target model
    to compute reducible loss for sample selection during training. It supports various ratio scheduling strategies
    for dynamic sample selection and handles model training and loading for specific datasets.

    Args:
        config (dict): Configuration dictionary containing method and dataset parameters.
            Expected keys include:
                - 'method_opt': Dictionary with keys 'ratio', 'budget', 'epochs', 'ratio_scheduler',
                  'warmup_epochs', 'iter_selection', 'balance'.
                - 'rho_loss': Dictionary with key 'training_budget'.
                - 'dataset': Dictionary with keys 'name' and 'num_classes'.
                - 'networks': Dictionary with key 'params' containing 'm_type'.
        logger (logging.Logger): Logger instance for logging training and selection information.
    """
    method_name = 'RhoLoss'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.setup_teacher_model(config, logger)

        # starting with uniform selection generally helps performance
        self.uniform_epochs = config['method_opt']['uniform_epochs'] if 'uniform_epochs' in config['method_opt'] else 0

        self.use_scores_as_distribution = config['method_opt'].get('scores_as_distribution', False)
        if self.use_scores_as_distribution:
            self.softmax_lambda = config['method_opt']['softmax_lambda']

    def setup_teacher_model(self, config, logger):
        """Retrieve the teacher model from config for computing irreducible loss."""
        teacher_config = dict(config)
        teacher_config['classes'] = self.data_info.get('classes')
        teacher_config['template'] = self.data_info.get('template')
        self.teacher_model = build_teacher_model(teacher_config, logger)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()        
    

    def compute_losses(self, inputs, targets):
        """Precompute irreducible losses for the training dataset using the teacher model."""
        self.teacher_model.eval()

        with torch.no_grad():
            outputs = self.teacher_model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='none')
            return loss.float()

    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            self.logger.info('warming up')
            return 1.0
        if self.ratio_scheduler == 'constant':
            return self.ratio
        elif self.ratio_scheduler == 'increase_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'decrease_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'increase_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        elif self.ratio_scheduler == 'decrease_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError

    def reducible_loss_selection(self, inputs, targets, indexes, number_to_select, epoch):
        """Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        # Set models to eval mode
        self.model.eval()

        # Get student loss from main model, irreducible loss from teacher model, and reducible loss by calculating the difference
        with torch.no_grad():
            total_loss = F.cross_entropy(self.model(inputs), targets, reduction='none')
        irreducible_loss = self.compute_losses(inputs, targets)
        reducible_loss = total_loss - irreducible_loss

        # Select samples with highest reducible loss
        if self.use_scores_as_distribution:
            weights = torch.softmax(self.softmax_lambda * reducible_loss, dim=0)
            index_selected = torch.multinomial(weights, number_to_select)
        else: # Normal RhoLoss
            _, index_selected = torch.topk(reducible_loss, k=number_to_select, largest=True, sorted=False)
        
        # Override with uniform selection if specified
        if epoch < self.uniform_epochs:
            self.logger.info('Uniform selection')
            index_selected = torch.randperm(len(inputs))[:number_to_select]
        
        # Return to train mode and return selected indices
        self.model.train()
        return index_selected.cpu().numpy(), reducible_loss.detach().cpu()

    def before_batch(self, i, inputs, targets, indexes):
        """Prepare the batch for training by selecting samples based on reducible loss.
        Args:
            i (int): Current batch index.
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
            indexes (torch.Tensor): Indices of the samples in the current batch.
        Returns:
            tuple: Selected inputs, targets, and indexes for the current batch.
        """
        # Get the ratio for the current epoch
        ratio = self.get_ratio_per_epoch(self._current_epoch)
        if ratio == 1.0:
            if i == 0:
                self.logger.info('using all samples')
            return super().before_batch(i, inputs, targets, indexes)
        else:
            if i == 0:
                self.logger.info(f'balance: {self.balance}')
                self.logger.info('selecting samples for epoch {}, ratio {}'.format(self._current_epoch, ratio))

        # Get indices based on reducible loss
        number_to_select = max(1, int(inputs.shape[0] * ratio))
        indices, scores = self.reducible_loss_selection(inputs, targets, indexes, number_to_select, self._current_epoch)
        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]
        return MinibatchInfo(inputs, targets, indexes, scores=scores)
