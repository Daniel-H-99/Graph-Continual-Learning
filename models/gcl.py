import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from models import register_model
import utils


@register_model("gcl")
class GCL(nn.Module):
    def __init__(
        self,
        args,
        in_channels=1,
        num_filters=64,
        num_classes=10,
        hidden_size=256,
        buffer_size=200,
        context_lambda=1.0,
        graph_lambda=10,
        context_temperature=2.0,
        target_temperature=2.0,
    ):
        super().__init__()
        self.in_channels, self.num_classes = in_channels, num_classes
        self.context_lambda, self.graph_lambda = context_lambda, graph_lambda
        self.context_temperature = context_temperature
        self.target_temperature = target_temperature
        self.args = args
        self.hidden_size = hidden_size

        if in_channels == 1:
            self.image_encoder = nn.Sequential(nn.Linear(784, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        else:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(num_filters,  num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
            )

        self.latent_mapping = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.edge_mapping = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.edge_scaling = nn.Parameter(torch.tensor(1.0 * hidden_size).sqrt().log())
        self.label_encoder = nn.Linear(num_classes, hidden_size)
        self.output_mapping = nn.Sequential(nn.ReLU(), nn.Linear(2 * hidden_size, num_classes))

        self.buffer_size, self.total_seen = buffer_size, 0
        self.image_buffer, self.label_buffer = [], []
        self.loss_buffer = np.full((buffer_size), float("inf"), dtype="float32")

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    @classmethod
    def build_model(cls, args):
        return cls(
            in_channels=args.in_channels,
            hidden_size=args.hidden_size,
            buffer_size=args.buffer_size,
            context_lambda=args.context_lambda,
            graph_lambda=args.graph_lambda,
            num_classes=args.num_classes,
            context_temperature=args.context_temperature,
            target_temperature=args.target_temperature,
            args=args,
        )

    def forward(self, target_images, target_labels=None, **kwargs):
        if len(self.image_buffer) == 0:
            target_embeddings = self.image_encoder(target_images)
            target_edges = self.compute_edge_logits(target_embeddings, target_embeddings)
            self.update_memory(target_images, target_labels, target_edges)
            return

        context_images, context_labels, context_losses = self.aggregate_context() # Context images and labels from Buffer
        num_contexts, num_targets = len(context_images), len(target_images)

        image_embeddings = self.image_encoder(torch.cat([context_images, target_images], dim=0))
        context_embeddings, target_embeddings = image_embeddings.split([num_contexts, num_targets], dim=0)

        labels = torch.cat([context_labels, target_labels], dim=0)

        context_edges = self.compute_edge_logits(image_embeddings, context_embeddings) # Returns logits for sampling G and A
        temperature = torch.tensor([self.context_temperature] * num_contexts + [self.target_temperature] * num_targets).cuda().view(-1, 1) # context: 5, target: 1
        context_dists = D.relaxed_bernoulli.LogitRelaxedBernoulli(logits=context_edges, temperature=temperature) # G and A
        
        
        unnorm_graph = context_dists.rsample().sigmoid()
        ### detereministic label ###
        # unnorm_graph = (context_edges.sigmoid() > 0.5).long().float()
        unnorm_graph.data.fill_diagonal_(0.0) # 0 for diag(G)
        norm_graph = F.normalize(unnorm_graph, p=1, dim=1) # Normalized G and A


        # TODO 1: Create final embedding
        # norm_graph: (n_c + n_t) x n_c
        # image_embeddings: (n_c + n_t ) x d_h
        # context_labels: n_c x num_classes

        def build_one_hot_vectors(class_indice):
          # class_indice: n_c
          result = torch.zeros(len(class_indice), self.num_classes)
          result[class_indice] = 1
          return result
        

        ### option: Add self loop for graph A ###
        # label_embeddings = self.label_encoder(build_one_hot_vectors(context_labels).cuda()); # n_c x d_h
        # context_classes = build_one_hot_vectors(context_labels).cuda()
        # target_classes = torch.mm(norm_graph[num_contexts:], context_classes)
        # classes = torch.cat([context_classes, target_classes], dim=0)
        # label_embeddings = self.label_encoder(classes)
        # context_label_embeddings, target_label_embeddings = label_embeddings.split([num_contexts, num_targets], dim=0)
        # v_c = torch.cat([self.latent_mapping(context_embeddings), context_label_embeddings], dim=1) # n_c x 2 * d_h
        # v_t = torch.cat([self.latent_mapping(target_embeddings), target_label_embeddings], dim=1)
        # final_embeddings_c = torch.mm(norm_graph[:num_contexts], v_c) # (n_c + n_t) x 2 * d_h
        # unnorm_graph_t = torch.cat([unnorm_graph[-num_targets:], torch.eye(num_targets).cuda()], dim=1)
        # norm_graph_t = F.normalize(unnorm_graph_t, p=1, dim=1)
        # final_embeddings_t = torch.mm(norm_graph_t, torch.cat([v_c, v_t], dim=0))
        # final_embeddings = torch.cat([final_embeddings_c, final_embeddings_t], dim=0)
        # logits = self.output_mapping(final_embeddings); # (n_c + n_t) x num_classes
        # context_logits, target_logits = logits.split([num_contexts, num_targets], dim=0)
        
        v_c = torch.cat([self.latent_mapping(context_embeddings), self.label_encoder(build_one_hot_vectors(context_labels).cuda())], dim=1)
        final_embeddings = torch.mm(norm_graph, v_c)
        logits = self.output_mapping(final_embeddings); # (n_c + n_t) x num_classes
        context_logits, target_logits = logits.split([num_contexts, num_targets], dim=0)

        context_loss = F.cross_entropy(input=context_logits, target=context_labels, reduction="none")
        target_loss = F.cross_entropy(input=target_logits, target=target_labels, reduction="none")
        current_losses = context_loss.detach().cpu().numpy()
        context_masks = context_losses < current_losses
        per_updated = context_masks.sum()/np.ones_like(context_masks).sum()
        self.loss_buffer[:num_contexts] = np.minimum(context_losses, current_losses)

        graph_loss = torch.tensor(0.0).cuda()
        if hasattr(self, "context_edges") and context_masks.sum() > 0 and self.graph_lambda > 0:
            H, W = self.context_edges.shape
            current_edges = context_edges[:H, :W][np.ix_(context_masks, context_masks)].sigmoid() # Indicate old context edges with lowest loss to be changed
            prev_edges = self.context_edges[np.ix_(context_masks, context_masks)].sigmoid()
            graph_loss = F.binary_cross_entropy(current_edges, prev_edges.detach()) # Graph regularization for context edges with lowest loss
            graph_loss = F.binary_cross_entropy(context_edges[:H, :W].sigmoid(), self.context_edges.sigmoid().detach())
        target_edges = self.compute_edge_logits(target_embeddings, target_embeddings)
        
        loss = target_loss.mean() + self.context_lambda * context_loss.mean() + self.graph_lambda * graph_loss 
        
        # TODO 2: Complete self.update_memory()
        self.update_memory(target_images, target_labels, target_edges, context_edges, context_losses < current_losses)
        

        return {
            "loss": loss,
            "context_loss": context_loss.mean(),
            "target_loss": target_loss.mean(),
            "graph_loss": graph_loss,
            "context_dists": context_dists,
            "context_images": context_images,
            "context_labels": context_labels,
            "context_masks": context_masks,
            "target_images": target_images,
            "target_labels": target_labels,
            "context_acc": context_logits.argmax(dim=-1).eq(context_labels).sum().float() / len(context_images),
            "image_embeddings": image_embeddings,
            "final_embeddings": final_embeddings,
            "per_updated": per_updated,
            "graphs": norm_graph.split([num_contexts, num_targets]),
        }
        
    def predict(self, target_images, target_labels, num_samples=30, **kwargs):
        context_images, context_labels, _ = self.aggregate_context()

        num_contexts, num_targets = len(context_images), len(target_images)
        image_embeddings = self.image_encoder(torch.cat([context_images, target_images], dim=0))
        context_embeddings, target_embeddings = image_embeddings.split([num_contexts, num_targets], dim=0)

        context_edges = self.compute_edge_logits(target_embeddings, context_embeddings)
        context_dists = D.relaxed_bernoulli.LogitRelaxedBernoulli(logits=context_edges, temperature=0.1)
        # context_dists = D.bernoulli(logits=context_edges)

        log_probs = target_images.new(target_images.size(0), self.num_classes, num_samples)
        for idx in range(num_samples):
            target_graph = F.normalize(context_dists.sample().sigmoid(), dim=1, p=1)
            context_latents = self.latent_mapping(context_embeddings)
            label_embeddings = self.label_encoder(F.one_hot(context_labels, self.num_classes).float())
            final_embeddings = torch.mm(target_graph, torch.cat([context_latents, label_embeddings], dim=-1))
            log_probs[:, :, idx] = F.log_softmax(self.output_mapping(final_embeddings), 1)

        log_probs = torch.logsumexp(log_probs, 2) - np.log(num_samples)

        return {
            "context_images": context_images,
            "context_labels": context_labels,
            "target_images": target_images,
            "target_labels": target_labels,
            "context_dists": context_dists,
            "final_embeddings": final_embeddings,
            "preds": log_probs.argmax(dim=-1),
            "target_graph": target_graph,
        }

    def update_memory(self, target_images, target_labels, target_edges, context_edges=None, context_masks=None):
    
        """[summary]
        Updates self.image_buffer, self.label_buffer, self.context_edges
        
        Params:
            target_images: current batch's new images
            target_labels: current batch's new labels
            target_edges: current batch's target to target logits (used for updating self.context_endges)
            context_edges: current batch's logits of G (context to context) and A (target to context)
            context_masks: Indicates context that reached the lowest loss

        self.image_buffer: Context images (Until previous batch)
        self.label_buffer: Context labels (Until previous batch)
        self.loss_buffer: Previous batch's context loss (Until previous batch)
        self.context_edges: Logits of G (Until previous batch / updated when the context reached the lowest)

        new_indices: buffer indice of the newly added
        target_indices: target indices of the newly added
        new_edges : contains the newly chosen target edges, unchanged context edges and changed context edges
        """

        new_indices, target_indices = [], []

        #* Reservoir Sampling
        for target_index, (target_image, target_label) in enumerate(zip(target_images, target_labels)):
            if len(self.image_buffer) < self.buffer_size:
                self.image_buffer.append(target_image)
                self.label_buffer.append(target_label)
                new_indices.append(len(self.image_buffer) - 1)
                target_indices.append(target_index)
            else:
                context_index = random.randrange(self.total_seen)
                if context_index < self.buffer_size:
                    self.image_buffer[context_index] = target_image
                    self.label_buffer[context_index] = target_label
                    self.loss_buffer[context_index] = float("inf")
                    new_indices.append(context_index)
                    target_indices.append(target_index)
            self.total_seen += 1

        indices = dict(zip(new_indices, target_indices))
        new_indices, target_indices = list(indices.keys()), list(indices.values())

        num_new_contexts = len(self.image_buffer) # Size of the updated buffer
        new_edges = target_images.new_zeros(num_new_contexts, num_new_contexts) # Initialize new buffer

        if context_edges is not None:
            num_targets = len(target_images)
            num_contexts = len(context_edges) - num_targets
            old_indices = np.array(list(set(range(num_new_contexts)) - set(new_indices))) # (buffer indices) \ (new target indices)
            
            # Update newly added context to target / target to context edges
            new_edges[np.ix_(old_indices, new_indices)] = context_edges[num_contexts:].T[np.ix_(old_indices, target_indices)].clone().detach()
            new_edges[np.ix_(new_indices, old_indices)] = context_edges[num_contexts:][np.ix_(target_indices, old_indices)].clone().detach()
            
            # TODO 2: Update context to context that reached the lowest loss (from current batch's context_edges to new_edges) 
            assert context_masks is not None
            context_masks_2d = torch.from_numpy(context_masks[old_indices]).cuda().unsqueeze(1).expand(-1, len(old_indices))
            new_edges[np.ix_(old_indices, old_indices)] = torch.where(context_masks_2d, context_edges[np.ix_(old_indices, old_indices)].clone().detach(), self.context_edges[np.ix_(old_indices, old_indices)].clone().detach())
        # Update newly added target to target edges
        new_edges[np.ix_(new_indices, new_indices)] = target_edges[np.ix_(target_indices, target_indices)].clone().detach()
        self.context_edges = new_edges
        
    def aggregate_context(self):
        if len(self.image_buffer) > 0:
            context_images = torch.stack(self.image_buffer, dim=0)
            context_labels = torch.stack(self.label_buffer, dim=0)
            context_losses = self.loss_buffer[: len(context_images)].copy()
            return context_images, context_labels, context_losses


    def compute_edge_logits(self, z1, z2):
        z1, z2 = self.edge_mapping(z1), self.edge_mapping(z2)
        distances = (z1 ** 2).sum(dim=1, keepdim=True) + (z2 ** 2).sum(dim=1) - 2 * z1 @ z2.T
        distances = -0.5 * distances / self.edge_scaling.exp()
        logits = utils.logitexp(distances.view(len(z1), len(z2)))
        return logits
