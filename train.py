import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys,time
import torch
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models
from test import Test
from os.path import join
from function import get_dataloader, train, val

def main():
    setpu_seed(2022)
    args = parse_args()
    os.chdir(args.now_dir)
    save_path = join(args.outf, args.save)
    save_args(args,save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')
    
    net = models.UNetFamily.AttU_Net(1,2).to(device)

    print("Total number of parameters: " + str(count_parameters(net)))

    # log.save_graph(net,torch.randn((1,1,48,48)).to(device).to(device=device))  # Save the model structure to the tensorboard file
    criterion = CrossEntropyLoss2d() # Initialize loss function
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    
    train_loader, val_loader = get_dataloader(args) # create dataloader
    
    if args.val_on_test: 
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = Test(args) 

    best = {'epoch':0,'AUC_roc':0.5} # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
        # train stage
        train_log = train(train_loader,net,criterion, optimizer,device) 
        # val stage
        if not args.val_on_test:
            val_log = val(val_loader,net,criterion,device)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()

        log.update(epoch,train_log,val_log) # Add log information
        lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_auc_roc'] > best['AUC_roc']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'],best['AUC_roc']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
if __name__ == '__main__':
    cudnn.benchmark = True
    main()
