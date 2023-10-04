import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import pi



class giraffeInsDecoder(nn.Module):
    ''' GIRAFFE isnatce Decoder class.
    semantic_aware decoder
    '''

    def __init__(self, 
    z_dim,
    use_semantic_aware = False, 
    semantic_list = ['car', 'building'],
    out_channel = 64,
    use_viewdirs = False,
    giraffe_decoder_kwargs = {},
    **kwargs):
        super().__init__()
        self.use_semantic_aware = use_semantic_aware
        self.semantic_list = semantic_list
        self.feature_dim = out_channel
        giraffe_decoder_kwargs.out_channel = out_channel

        if use_semantic_aware:
            for s in semantic_list:
                if s == 'car':
                    self.nerf_car = giraffeDecoder(z_dim = z_dim,use_viewdirs =use_viewdirs, **giraffe_decoder_kwargs)
                if s == 'building':
                    self.nerf_building = giraffeDecoder(z_dim = z_dim,
                    use_viewdirs = use_viewdirs,**giraffe_decoder_kwargs)
        else: 
            self.nerf = giraffeDecoder(z_dim = z_dim,use_viewdirs = use_viewdirs,**giraffe_decoder_kwargs)


    def forward(self, z, pts, c, semantic, ray_d, **kwargs):
        if self.use_semantic_aware:
            assert semantic != None
            p_num = z.shape[0]
            feat_out, sigma_out = torch.zeros(p_num, self.feature_dim).to(z.device),torch.zeros(p_num, 1).to(z.device)
            for s in self.semantic_list:
                if s == 'car':
                    car_idx = torch.any(semantic == 26, dim = 1)
                    feat_out[car_idx], sigma_out[car_idx] = self.nerf_car(pts = pts[car_idx], z = z[car_idx],ray_d = ray_d[car_idx], c = c[car_idx])
                if s == 'building':
                    building_idx = torch.any(semantic == 11, dim = 1)
                    feat_out[building_idx], sigma_out[building_idx] = self.nerf_building(pts = pts[building_idx], z = z[building_idx],ray_d = ray_d[building_idx], c = c[building_idx])
        else:
            feat_out, sigma_out = self.nerf(pts = pts, z = z,ray_d = ray_d, c = c)

        return feat_out, sigma_out


class giraffeDecoder(nn.Module):
    ''' Decoder class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc_pts (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''

    def __init__(self, 
        hidden_size=128, 
        n_blocks=4, 
        n_blocks_view=1,
        skips=[2], 
        use_viewdirs=True, 
        n_freq_posenc_pts=10,
        n_freq_posenc_views=4,
        n_freq_posenc_c=4,
        z_dim = 64, 
        c_dim = 3,
        out_channel = 16,
        out_channels_c=3, 
        final_sigmoid_activation=False,
        downscale_p_by=0.5, 
        positional_encoding="normal",
        gauss_dim_pos=10, 
        gauss_dim_view=4, 
        gauss_std=4.,
        **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc_pts = n_freq_posenc_pts
        self.n_freq_posenc_views = n_freq_posenc_views
        self.n_freq_posenc_c = n_freq_posenc_c
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.use_final_sigmoid = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            pts_dim_embed = 3 * gauss_dim_pos * 2
            view_dim_embed = 3 * gauss_dim_view * 2
        else:
            pts_dim_embed = 3 * self.n_freq_posenc_pts * 2
            c_dim_embed = 3 * self.n_freq_posenc_c * 2
            view_dim_embed = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.fc_in = nn.Linear(pts_dim_embed, hidden_size)

        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        if c_dim > 0:
            self.fc_c = nn.Linear(c_dim_embed, hidden_size)


        blocks = [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)
        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(pts_dim_embed, hidden_size) for i in range(n_skips)
            ])
            self.fc_c_skips = nn.ModuleList([
                nn.Linear(c_dim_embed, hidden_size) for i in range(n_skips)
            ])
        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(view_dim_embed, hidden_size)
        self.feat_out = nn.Linear(hidden_size, out_channel)
        self.rgb_out = nn.Linear(out_channel, out_channels_c)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [nn.Linear(view_dim_embed + hidden_size, hidden_size)
                 for i in range(n_blocks_view - 1)])

    def transform_points(self, p, L = 4):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by
        # a = p.detach().cpu().numpy()

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = L
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, pts,z, c = None, ray_d = None,**kwargs):
        z_shape = z_app = z
        a = F.relu
        if self.z_dim > 0:
            batch_size = pts.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(pts.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(pts.device)

        p = self.transform_points(pts, L = self.n_freq_posenc_pts)
        net = self.fc_in(p)

        if z_shape is not None:
            net = net + self.fc_z(z_shape) # work for ray batch
        if self.c_dim > 0:
            c = self.transform_points(c, L = self.n_freq_posenc_c)
            net = net + self.fc_c(c)
            #net = net + self.fc_z(z_shape).unsqueeze(1) # work for bbx batch
        net = a(net)

        skip_idx = 0
        #net = a(self.blocks(net))
        for idx, layer in enumerate(self.blocks):
            net = a(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape)
                net = net + self.fc_p_skips[skip_idx](p)
                if self.c_dim > 0:
                    net = net + self.fc_c_skips[skip_idx](c)
                skip_idx += 1
        sigma = self.sigma_out(net).squeeze(-1)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app)
        if self.use_viewdirs and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, L = self.n_freq_posenc_views)
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat = self.feat_out(net)

        if True: 
            feat = torch.tanh(feat)

        # rgb = self.rgb_out(feat)
        # #sigma = torch.sigmoid(sigma)
        # if self.use_final_sigmoid:
        #     rgb= torch.sigmoid(rgb)
        # feat = torch.cat((feat, rgb), dim = -1)
        return feat ,sigma.unsqueeze(-1)




class giraffeDecoder2D(nn.Module):
    def __init__(self, 
        hidden_size=128, 
        n_blocks=4, 
        n_blocks_view=1,
        skips=[2], 
        use_viewdirs=True, 
        n_freq_posenc_pts=10,
        n_freq_posenc_views=4,
        z_dim=64, 
        rgb_out_dim=3, 
        out_channel = 16,
        out_channels_c=3, 
        final_sigmoid_activation=False,
        downscale_p_by=0.5, 
        positional_encoding="normal",
        gauss_dim_pos=10, 
        gauss_std=4.,
        **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc_pts = n_freq_posenc_pts
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            pts_dim_embed = 3 * gauss_dim_pos * 2
            view_dim_embed = 3 * gauss_dim_view * 2
        else:
            pts_dim_embed = 3 * self.n_freq_posenc_pts * 2
            pts_dim_embed = 3 * self.n_freq_posenc_c * 2
            view_dim_embed = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.fc_in = nn.Linear(pts_dim_embed, hidden_size)
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)


        blocks = [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)
        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(pts_dim_embed, hidden_size) for i in range(n_skips)
            ])
        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(view_dim_embed, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [nn.Linear(view_dim_embed + hidden_size, hidden_size)
                 for i in range(n_blocks_view - 1)])

    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by
        # a = p.detach().cpu().numpy()

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc_pts
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, pts, z,ray_d = None,**kwargs):
        z_shape = z_app = z
        a = F.relu
        if self.z_dim > 0:
            batch_size = pts.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(pts.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(pts.device)
        p = self.transform_points(pts)
        net = self.fc_in(p)
        #a_debug = self.fc_z(z_shape)
        if z_shape is not None:
            net = net + self.fc_z(z_shape) # work for ray batch
            #net = net + self.fc_z(z_shape).unsqueeze(1) # work for bbx batch
        net = a(net)

        skip_idx = 0
        #net = a(self.blocks(net))
        for idx, layer in enumerate(self.blocks):
            net = a(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        # sigma_out = self.sigma_out(net).squeeze(-1)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app)
        if self.use_viewdirs and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, views=True)
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return rgb_out
