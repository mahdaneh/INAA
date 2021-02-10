import os
import torch

import pretrainedmodels
import pretrainedmodels.utils as utils
from torchvision import transforms
import torchvision
import foolbox
from torch.utils.data import DataLoader
import load_data as ldb
import numpy as np
import tqdm
import foolbox.attacks as Fattacks
from collections import defaultdict
import utils as local_util
import pickle

def foolbox_attacks (gen_model, disc_model, data_loader_recon, data_loader_orig,device):
    print(len(data_loader_orig), len(data_loader_recon))
    assert len(data_loader_orig)==len(data_loader_recon)

    _adv_perclass =50

    gen_model.eval()
    disc_model.eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    std = torch.tensor(preprocessing['std'],dtype=torch.float, device=device)
    mean = torch.tensor(preprocessing['mean'], dtype=torch.float, device=device)
    attack_model = foolbox.models.PyTorchModel(gen_model, bounds=(0, 1), preprocessing=preprocessing)
    attacks = {
        'FGS':Fattacks.FGSM()}
              #     'PGD':Fattacks.LinfPGD(),
              # 'BIM':Fattacks.LinfBasicIterativeAttack()}

    epsilons = [0.001, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05 ]
    print("epsilons")
    print(epsilons)


    attack_success = np.zeros((len(attacks), len(epsilons), len(data_loader_orig)), dtype=np.bool)


    for i, (attack_name,attack) in enumerate(attacks.items()):


        print (' %s' % attack_name +''*10+'epsilons '+str(epsilons))
        d = defaultdict(list)
        d['name']= attack_name
        d['true label']=[]

        counter = 0

        for (img_rec, labels_rec), (img_orig, labels_orig) in zip(data_loader_recon,data_loader_orig):

            counter +=1
            # torchvision.utils.save_image(img_orig,'%d_o.png'%counter)
            # torchvision.utils.save_image(img_rec, '%d_r.png' % counter)

            img_orig = img_orig.to(device)
            img_rec = img_rec.to(device)
            labels_rec = labels_rec.to(device)
            labels_orig = labels_orig.to(device)
            disc_model = disc_model.to(device)
            gen_model = gen_model.to(device)



            orig_norm = img_orig.clone()
            # for k in range(3):
            #     img_rec[:,k,:,:] = (img_rec[:,k,:,:]-mean[k])/std[k]
            #     orig_norm[:, k, :, :] = (orig_norm[:, k, :, :] - mean[k]) / std[k]
            norm = transforms.Normalize(mean, std)
            orig_norm = norm(orig_norm)
            img_rec = norm(img_rec)
            disc_model.eval()
            gen_model.eval()
            clean_acc_rec = local_util.accuracy(disc_model(img_rec),labels_rec)[0]
            clean_acc_orig = local_util.accuracy(gen_model(orig_norm),labels_orig)[0]
            # only those reconstructed correctly classified and if correctly classified by the classifier


            if clean_acc_rec.item()!=0 and clean_acc_orig.item()!=0 and\
                    (len(d['true label'])==0 or d['true label'].count(labels_orig.cpu().data[0])<_adv_perclass ):


                _, clipped_advs, success = attack(attack_model, img_orig, labels_orig, epsilons=epsilons)

                fooling_class = [torch.argmax(attack_model(clipped_advs[i]),axis=1) for i in range(len(clipped_advs))]
                if success.any():
                    print ('number of samples %d'%len(d['true label']))


                    [d['advs%.4f'%eps].append([np.asarray(adv.cpu()),suc.cpu()]) for eps, adv, suc in zip(epsilons,clipped_advs,success)]

                    [d['fooling_label%.4f'%eps].append(fl_cls) for eps, fl_cls in zip(epsilons,fooling_class)]

                    d['cleans'] +=img_orig.data.cpu()
                    d['true label'] +=labels_orig.data.cpu()


                assert success.shape == (len(epsilons), len(img_orig))
                success_ = success.cpu().numpy()
                assert success_.dtype == np.bool
                attack_success[i] = success_

                # pbar.set_description('sucess' +str(1.0 - success_.mean(axis=-1))+' %d total advs \t'%len(d['cleans']) +\
                #                      '%d advs for class %d'%(d['true label'].count(labels.cpu().data[0]),labels.cpu().data[0]))



        # torch.save(d,file_path+str(attack_name)+'.pt')

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

    return d

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    gen_name = 'resnet101'
    gen_classifier, transform_nomean = local_util.load_model(arch=gen_name)

    orig_dataset = ldb.load_local_data(image_path='Clean_image_FGS.pt', label_path='clean_imagenette_label_FGS',
                                       transform=transform_nomean, convert_2_10cls=True if gen_name=='vgg' else False)
    orig_dataloader = DataLoader(orig_dataset, 1, shuffle=False, num_workers=0)

    discrm_classifier, nomean_transform = local_util.load_model(arch='resnet101')

    # HR images
    DT_name = ['HR_images/CAR_x4_clean_FGS']
    for name in DT_name:

        dataset = ldb.load_local_data(image_path=name, label_path='clean_imagenette_label_FGS', transform=nomean_transform)
        data_loader = DataLoader(dataset, 1, shuffle=False, num_workers=0)
        adversaries_data = foolbox_attacks(gen_classifier, discrm_classifier, data_loader, orig_dataloader, device=device)

    file_path = '%s_%d_%s' % (os.path.splitext(name)[0], 224, gen_name)

    with open(file_path + '.pickle', 'wb') as fp:
        pickle.dump(adversaries_data, fp, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    main()
