import torch
# from modules.models.inception import inception_v3
from modules.models.resnet_pytorch import resnet18
from modules.models.inception_pytorch import inception_v3

class ModelHandler:
    @staticmethod
    def save_checkpoint(state, filename):
        torch.save(state, filename)

    @staticmethod
    def get_new_model():
        model = inception_v3(pretrained=True)
        return model

    @staticmethod
    def load_optimizer(optimizer, checkpoint_path, gpu_mode):
        if gpu_mode:
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer'])

        return optimizer

    @staticmethod
    def load_model(model, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k
            if k[0:7] == 'module.':
                name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.cpu()

        return model

    @staticmethod
    def load_model_for_training(model, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        encoder_state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in encoder_state_dict.items():
            name = k
            if k[0:7] == 'module.':
                name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.cpu()

        return model
