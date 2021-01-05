import os
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
import foolbox
from torch.utils.data import DataLoader
import load_data as ldb
import numpy as np
import tqdm
import foolbox.attacks as Fattacks
from collections import defaultdict
import utils as util
import pickle
def load_adapt_imagenette_imagenet(batchsize=1, transform=None):
    data = ldb.ImageFolder_imagenette(root='data/imagenette',transform=transform)
    test_loader = DataLoader(data, batchsize, shuffle=False, num_workers=0)
    return test_loader



def load_model (arch='resnet101', pretrained = 'imagenet'):

    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
    import pdb;pdb.set_trace()
    if 'scale' in pretrainedmodels.pretrained_settings[arch][pretrained]:
        scale = pretrainedmodels.pretrained_settings[arch][pretrained]['scale']
    else:
        scale = 0.875

    transform = pretrainedmodels.utils.TransformImage(
        model,
        scale=scale,
        preserve_aspect_ratio=False)

    return  model, transform

def foolbox_attacks (model, data_loader, file_path,device):

    _adv_perclass = 100

    model.eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    attack_model = foolbox.models.PyTorchModel(model, bounds=(0, 1))
    attacks = {
        'FGS':Fattacks.FGSM(),
                  'PGD':Fattacks.LinfPGD(),
              'BIM':Fattacks.LinfBasicIterativeAttack(),
              'DeepFool':Fattacks.L2DeepFoolAttack(),
    }

    epsilons = [
        0.002,
        0.001,
        0.005]
    print("epsilons")
    print(epsilons)
    print("")

    attack_success = np.zeros((len(attacks), len(epsilons), len(data_loader)), dtype=np.bool)


    for i, (attack_name,attack) in enumerate(attacks.items()):


        print (' %s' % attack_name +''*10+'epsilons '+str(epsilons))
        d = defaultdict(list)
        d['name']= attack_name
        d['true label']=[]
        pbar = tqdm.tqdm(data_loader)

        for j , (images, labels) in enumerate(pbar):

            images = images.to(device)
            labels = labels.to(device)

            prec1, prec5 = util.accuracy(model(images).data,labels, topk=(1, 5))


            if prec1==100 and (len(d['true label'])==0 or d['true label'].count(labels.cpu().data[0])<_adv_perclass ):

                _, clipped_advs, success = attack(attack_model, images, labels, epsilons=epsilons)
                fooling_class = torch.argmax(model(torch.cat(clipped_advs,axis=0)),axis=1)

                [d['advs%.4f'%eps].append([adv,suc]) for eps, adv, suc in zip(epsilons,clipped_advs,success)]
                [d['fooling_label%.4f'%eps].append(fl_cls) for eps, fl_cls in zip(epsilons,fooling_class)]

                d['cleans'] +=images.data.cpu()
                d['true label'] +=labels.data.cpu()


                assert success.shape == (len(epsilons), len(images))
                success_ = success.cpu().numpy()
                assert success_.dtype == np.bool
                attack_success[i] = success_

                pbar.set_description('sucess' +str(1.0 - success_.mean(axis=-1))+' %d total advs \t'%len(d['cleans']) +\
                                     '%d advs for class %d'%(d['true label'].count(labels.cpu().data[0]),labels.cpu().data[0]))

        # torch.save(d,file_path+str(attack_name)+'')
        with open(file_path+str(attack_name)+'.pickle', 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked) using the best attack per sample
    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    print("")
    print("-" * 79)
    print("")
    print("worst case (best attack per-sample)")
    print("  ", robust_accuracy.round(2))
    print("")

    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked) using the best attack per sample

    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print (device)

    model, transform = load_model()

    # data_loader = load_adapt_imagenette_imagenet(batchsize=128, transform=transform)
    #
    # util.eval_accuracy_imgnet(model, data_loader,device)

    data_loader = load_adapt_imagenette_imagenet(batchsize=1, transform=transform)
    foolbox_attacks(model, data_loader, file_path='generated_attack_FoolBox', device=device)



if __name__ == '__main__':
    main()
