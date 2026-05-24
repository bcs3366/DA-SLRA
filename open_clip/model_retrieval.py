from .das import DAS_baseline
import open_clip
import torch
import torch.nn as nn



class OpenCLIPVisualEncoder(nn.Module):
    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual

    def forward(self, image, image_ori=None):
        return self.visual(image, image_ori)


class OpenCLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.register_buffer("attn_mask", clip_model.attn_mask, persistent=False)

    def forward(self, text_ids):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text_ids).to(cast_dtype)              # [B,L,C]
        x = x + self.positional_embedding.to(cast_dtype)               # [B,L,C]
        x = x.permute(1, 0, 2)                                         # [L,B,C]
        x = self.transformer(x, attn_mask=self.attn_mask)              # [L,B,C]
        x = x.permute(1, 0, 2)                                         # [B,L,C]
        x = self.ln_final(x)                                           # [B,L,C]
        eot = text_ids.argmax(dim=-1)                                  # [B]
        feat = x[torch.arange(x.size(0), device=x.device), eot] @ self.text_projection
        return feat

class DA_SLRA(DAS_baseline):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True, use_contrastive_loss=True, \
                         use_affil_loss=False)
        self.config = config
        self.use_affil_loss = config['use_affil_loss']
        self.use_triplet_loss = config['use_triplet_loss']
        self.align_before = False
        self.create_and_load_pretrained(config)

    def create_and_load_pretrained(self, config):
        if self.config['model'] == 'geo': 
            self.model, _ ,_ = open_clip.create_model_and_transforms("ViT-B/32", pretrained='openai')

            if self.config['if_evaluation'] == False:
                ckpt_path = "./models/pretrain/RS5M_ViT-B-32_RET-2.pt"
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            self.model, _, _ = open_clip.create_model_and_transforms("ViT-B/32")

    def get_vis_emb(self, image, image_ori = None, idx=None, label=None):
        if self.config['is_das']:
            if self.align_before:
                img_emb, feas_vis = self.model.encode_image(image, image_ori, normalize=True)
                return img_emb, feas_vis
            else:
                img_emb = self.model.encode_image(image, image_ori, normalize=True)
            return img_emb

    def get_vis_emb1(self, image, image_ori = None, idx=None, label=None):
        if self.config['is_das']:
            if self.align_before:
                img_emb, feas_vis = self.model.encode_image1(image, image_ori, normalize=True)
                return img_emb, feas_vis
            else:
                img_emb = self.model.encode_image1(image, image_ori)
            return img_emb

    def get_vis_emb_eval(self, image, idx=None, label=None):
        if self.config['is_das']:
            if self.align_before:
                img_emb, feas_vis = self.model.encode_image(image, normalize=True)
                return img_emb, feas_vis
            else:
                img_emb = self.model.encode_image(image, normalize=True)
            return img_emb
        
    def get_txt_emb(self, text_ids, idx=None, label=None):
        if self.config['is_das']:
            if self.align_before:
                txt_emb, feas_txt = self.model.encode_text(text_ids, normalize=True)
                return txt_emb, feas_txt
            else:
                txt_emb = self.model.encode_text(text_ids, normalize=True)
            return txt_emb

    def forward(self, image, text_ids, image_ori = None, idx=None, label=None, return_feats=False):
        if self.config['is_das']:
            if self.align_before:
                img_emb, feas_vis = self.get_vis_emb(image, image_ori)
                txt_emb, feas_txt = self.get_txt_emb(text_ids)
            else:
                img_emb = self.get_vis_emb(image, image_ori)
                txt_emb = self.get_txt_emb(text_ids)

        else:
            img_emb = self.get_vision_fusion_embeds(image, image_ori, self.config)
            txt_emb = self.get_text_fusion_embeds(text_ids, image_ori, self.config)

        if self.use_affil_loss:
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_affil = self.get_affil_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            return loss_contr, loss_affil

        elif self.use_triplet_loss:
            loss_triplet = self.get_triplet_loss(img_emb, txt_emb)
            return loss_triplet
        else:
            loss_before_contr = []
            if self.align_before:
                for i in range(len(feas_vis)):
                    loss_contr = self.get_contr_loss(feas_vis[i], feas_txt[i], idx=idx, label=label, config=self.config)
                    loss_before_contr.append(loss_contr)
                total_loss_before = sum(loss_before_contr)
            loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            if self.align_before:
                return loss_contr, loss_triplet, total_loss_before
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            loss_rec = self.rec_loss(img_emb, txt_emb)
            if return_feats:
                scores = img_emb @ txt_emb.t()  # cosine (normalize=True)
                return loss_contr, loss_triplet, loss_rec,  img_emb, txt_emb, scores

            #TODO new loss
            return loss_contr, loss_triplet, loss_rec