from data.dataset import readIndex, MY_dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

video_path = r"C:\Users\Nimisha\Videos\20230123_181300.mp4"
print(video_path)


def test(test_data_path='data/test_example.txt',
         save_path='deepcrack_results/',
         pretrained_model='checkpoints/DeepCrack_CT260_FT1.pth', ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    ##num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    ##model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)
        
    model.load_state_dict(trainer.saver.load(pretrained_model, map_location=torch.device('cuda')))

    model.eval()
        
    #print(test_list, test_loader)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #out = cv2.VideoWriter("deepcrack_results\\output.avi", fourcc, 1, (512,256))
    frames = []
    #cap.set(3,512) # set Width
    #cap.set(4,512) # set Height
    success, frame = cap.read()
    count = 0
    while success:
        count += 1
        #success, frame = cap.read()
        frame=cv2.resize(frame, (256, 256))
        cv2.imshow("Hello", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
        test_pipline = MY_dataReadPip(img=frame, transforms=None)

        test_list = readIndex(test_data_path)

        test_dataset = loadedDataset(test_list, preprocess=test_pipline)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                shuffle=False, num_workers=1, drop_last=False)

        

        with torch.no_grad():
            for names, (img, lab) in tqdm(zip(test_list, test_loader)):
                print("NAMES", names, "\n\nIMG", img, "\n\nLAB", lab)
                test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(
                    device)
                test_pred = trainer.val_op(test_data, test_target)
                test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
                save_pred = torch.zeros((256 * 2, 256))
                save_pred[:256, :] = test_pred
                save_pred[256:, :] = lab.cpu().squeeze()
                save_name = os.path.join(save_path, os.path.split(names[1])[1])
                save_pred = save_pred.numpy() * 255
                cv2.imwrite("deepcrack_results\\" + str(count) + "frame.jpg", save_pred.astype(np.uint8))
                cv2.imshow("Camera", save_pred.astype(np.uint8))
                #out.write(save_pred.astype(np.uint8))
                frames += [save_pred.astype(np.uint8)]
                #cv2.imshow("Camera", test_pred.numpy().astype(np.uint8)*255)
                k = cv2.waitKey(30) & 0xff
                if k == 27: # press 'ESC' to quit
                    break
                print("Frame done")
        print("DONE")
        success, frame = cap.read()
    #cap.release()
    #out.release()
    #out = cv2.VideoWriter(r'C:\Users\Dell\Desktop\Python\PyTorch\MachineLearning\CrackDetection\DeepCrack-master\DeepCrack-master\codes - Copy (2)\video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (frames[0].shape[0],frames[0].shape[1]))
    #for x in frames:
    #    out.write(x)
    #out.release()
    #fig = plt.figure()
    #ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    #ani.save('movie.mp4')


if __name__ == '__main__':
    test()
    #input()
