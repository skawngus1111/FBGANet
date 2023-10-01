import os
import sys

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.get_functions import get_save_path
def plot_loss_acc(history, model_dirs):
    params = yaml.safe_load(open(f'configuration_files/plot_configurations/plot_loss_configuration.yml'))

    train_loss, test_loss = history['train_loss'], history['val_loss']
    # train_top1_acc, test_top1_acc = history['train_top1_acc'], history['test_top1_acc']
    # train_top5_acc, test_top5_acc = history['train_top5_acc'], history['test_top5_acc']

    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss', color=params['train_color'])
    plt.plot(np.arange(len(test_loss)), test_loss, label='val loss', color=params['test_color'])

    plt.xlim([np.arange(len(train_loss))[0], np.arange(len(train_loss))[-1]])
    plt.ylim(params['loss_range'])
    plt.legend(loc='upper right')
    plt.grid(params['y_grid'], axis='y', linestyle='--')

    plt.savefig(os.path.join(model_dirs, 'plot_results/loss.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plt.plot(np.arange(len(train_top1_acc)), train_top1_acc, label='train top1 accuracy', color=params['train_color'])
    # plt.plot(np.arange(len(test_top1_acc)), test_top1_acc, label='test top1 accuracy', color=params['test_color'])
    #
    # plt.xlim([np.arange(len(train_top1_acc))[0], np.arange(len(train_top1_acc))[-1]])
    # plt.ylim(params['acc_range'])
    # plt.legend(loc='lower right')
    # plt.grid(params['y_grid'], axis='y', linestyle='--')
    #
    # plt.savefig(os.path.join(model_dirs, 'plot_results/top1_accuracy.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
    # plt.close()
    #
    # plt.plot(np.arange(len(train_top5_acc)), train_top5_acc, label='train top5 accuracy', color=params['train_color'])
    # plt.plot(np.arange(len(test_top5_acc)), test_top5_acc, label='test top5 accuracy', color=params['test_color'])
    #
    # plt.xlim([np.arange(len(train_top5_acc))[0], np.arange(len(train_top5_acc))[-1]])
    # plt.ylim(params['acc_range'])
    # plt.legend(loc='lower right')
    # plt.grid(params['y_grid'], axis='y', linestyle='--')
    #
    # plt.savefig(os.path.join(model_dirs, 'plot_results/top5_accuracy.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
    # plt.close()

def plot_biomedical_image_and_prediction(args, image, target, predict, count):
    model_dirs = get_save_path(args)

    if not os.path.exists(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type))):
        os.makedirs(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type)))

    image = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    target = target.squeeze().cpu().detach().numpy()
    predict = predict.squeeze().cpu().detach().numpy() >= 0.5

    misprediction_map = np.abs(target - predict)
    misprediction_map = np.expand_dims(misprediction_map, axis=2)
    misprediction_map = np.repeat(misprediction_map, 3, axis=2)
    misprediction_map[np.mean(misprediction_map, axis=2) != 0] = (1, 0, 0)

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(image)
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].title.set_text('Input Image')

    ax[1].imshow(target, cmap='gray')
    ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].title.set_text('Ground Truth')

    ax[2].imshow(predict, cmap='gray')
    ax[2].axis('off'); ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].title.set_text('Prediction')

    ax[3].imshow(misprediction_map)
    ax[3].axis('off'); ax[3].set_xticks([]); ax[3].set_yticks([])
    ax[3].title.set_text('Misprediction Map')

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type), 'example_{}.png'.format(count)),
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # if not os.path.exists(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type), 'PRED_Boundary')):
    #     os.makedirs(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type), 'PRED_Boundary'))
    # if not os.path.exists(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type), 'GT_Boundary')):
    #     os.makedirs(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type), 'GT_Boundary'))
    #
    # target = np.array(target * 255, dtype=np.uint8)
    # predict = np.array(predict * 255, dtype=np.uint8)
    #
    # RGBforLabel = {'PRED': (255, 0, 0), 'GT': (0, 255, 0)}
    # contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours_predict, _ = cv2.findContours(predict, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # # Iterate over all contours
    # mask = np.zeros((args.image_size, args.image_size, 3), np.uint8)
    # for i, c in enumerate(zip(contours)):
    #     cv2.drawContours(mask, [c], -1, RGBforLabel.get('GT'), 2)
    #
    # pred_mask = np.zeros((args.image_size, args.image_size, 3), np.uint8)
    # for i, c in enumerate(contours_predict):
    #     colour = RGBforLabel.get('PRED')
    #     cv2.drawContours(pred_mask, [c], -1, colour, 2)
    #
    # PRED_Boundary = predict.reshape(args.image_size, args.image_size, -1) + pred_mask
    # plt.imshow(PRED_Boundary)
    # plt.axis('off'); plt.xticks([]); plt.yticks([])
    # plt.tight_layout()
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    # plt.savefig(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type), 'PRED_Boundary', 'PRED_Boundary_{}.png'.format(count)),
    #             bbox_inches='tight', pad_inches=0, dpi=100)
    # plt.close()
    #
    # plt.imshow(mask)
    # plt.axis('off'); plt.xticks([]); plt.yticks([])
    # plt.tight_layout()
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    # plt.savefig(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization({}->{})'.format(args.train_data_type, args.test_data_type), 'GT_Boundary', 'GT_Boundary_{}.png'.format(count)),
    #             bbox_inches='tight', pad_inches=0, dpi=100)
    # plt.close()

def plot_forest_image_and_prediction(args, image, target, predict, count) :
    model_dirs = get_save_path(args)

    if not os.path.exists(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization', 'DenseCRF:{}'.format(args.crf))):
        os.makedirs(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization', 'DenseCRF:{}'.format(args.crf)))

    IDX_TO_COLOR = ForestImageSegmentationDataset.IDX_TO_COLOR
    IDX_TO_NAME = ForestImageSegmentationDataset.IDX_TO_NAME

    patches = [mpatches.Patch(color=IDX_TO_COLOR[i], label=IDX_TO_NAME[i]) for i in IDX_TO_COLOR]

    image = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))
    target = target.cpu().detach().numpy()
    predict = predict.cpu().detach().numpy()

    new_target = np.zeros_like(image)
    for idx, color_map in IDX_TO_COLOR.items():
        idxx, idxy = np.where(target == idx)
        new_target[idxx, idxy] = color_map

    new_predict = np.zeros_like(image)
    for idx, color_map in IDX_TO_COLOR.items():
        idxx, idxy = np.where(predict == idx)
        new_predict[idxx, idxy] = color_map

    idxx, idxy = np.where(np.sum(new_target, axis=2) == 0)
    new_predict[idxx, idxy] = (0, 0, 0)

    misprediction_map = np.abs(new_target - new_predict)
    misprediction_map[np.sum(misprediction_map, axis=2) != 0] = (1, 0, 0)

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(image)
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].title.set_text('Input Image')

    ax[1].imshow(new_target)
    ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].title.set_text('Ground Truth')

    ax[2].imshow(new_predict)
    ax[2].axis('off'); ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].title.set_text('Prediction')

    ax[3].imshow(misprediction_map)
    ax[3].axis('off'); ax[3].set_xticks([]); ax[3].set_yticks([])
    ax[3].title.set_text('Misprediction Map')

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(os.path.join(model_dirs, 'plot_results', 'inference_result_visualization', 'DenseCRF:{}'.format(args.crf), 'example_{}.png'.format(count)),
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def plot_data_dictribution():
    CVC_ClinicDB = np.random.multivariate_normal([1, 1], [[0.5, 1], [0.5, 0.5]], 200)
    CVC_300 = np.random.multivariate_normal([3, 1], [[0.1, 0.1], [0.3, 0.1]], 100)
    Kvasir_SEG = np.random.multivariate_normal([2.5, -0.5], [[0.1, 0.01], [0.03, 0.01]], 150)
    CVC_ColonDB = np.random.multivariate_normal([2, -1.5], [[0.01, 0.01], [0.03, 0.01]], 50)
    ETIS = np.random.multivariate_normal([4, -1], [[0.01, 0.01], [0.03, 0.01]], 40)

    plt.scatter(CVC_ClinicDB[:, 0], CVC_ClinicDB[:, 1], color='k', s=5, label='CVC-ClinicDB')
    plt.scatter(CVC_300[:, 0], CVC_300[:, 1], color='g', s=5, label='CVC-300')
    plt.scatter(Kvasir_SEG[:, 0], Kvasir_SEG[:, 1], color='b', s=5, label='Kvasir-SEG')
    plt.scatter(CVC_ColonDB[:, 0], CVC_ColonDB[:, 1], color='r', s=5, label='CVC-ColonDB')
    plt.scatter(ETIS[:, 0], ETIS[:, 1], color='m', s=5, label='ETIS')
    plt.xticks([]); plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('data_distribution.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
if __name__=='__main__':
    # plot_data_dictribution()
    data = [23.77039909362793, 23.949567794799805, 23.97222328186035, 24.00444793701172, 24.01193618774414, 24.156383514404297, 24.172319412231445, 24.206880569458008, 24.234655380249023, 24.28214454650879, 24.368192672729492, 24.442880630493164, 24.553119659423828, 24.629472732543945, 24.669504165649414, 25.130815505981445, 25.131839752197266, 25.144384384155273, 25.150175094604492, 25.154016494750977, 25.1592960357666, 25.164352416992188, 25.167743682861328, 25.213760375976562, 25.234527587890625, 25.241535186767578, 25.245824813842773, 25.249919891357422, 25.25187110900879, 25.256128311157227, 25.257535934448242, 25.260671615600586, 25.26300811767578, 25.268640518188477, 25.28278350830078, 25.293920516967773, 25.297504425048828, 25.30441665649414, 25.30841636657715, 25.317855834960938, 25.335487365722656, 25.353471755981445, 25.357343673706055, 25.378591537475586, 25.387487411499023, 25.459104537963867, 25.477407455444336, 25.489120483398438, 25.5032958984375, 25.5383358001709, 25.58515167236328, 25.631263732910156, 25.646623611450195, 25.65564727783203, 25.660512924194336, 25.846752166748047, 26.62531280517578, 28.307743072509766, 28.405759811401367, 28.43903923034668, 28.845792770385742]
    print(np.mean(data))