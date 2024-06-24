import logging

import torch

from utils.tool_utils import set_seed, set_logger, preprocess_graph

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score 
from config import parser
import time

import optimizers

import dgl
import pickle
import datetime
import warnings
from models.architecture import SHAN


warnings.filterwarnings("ignore")


if __name__ == '__main__':
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parser.parse_args()
    set_logger('./exp.log')
    logging.info(args)
    if args.cuda != -1 :
        device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    start_time_process = time.time()
    
    with open('../data/'+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('../data/'+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    
    with open('../data/'+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)


    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor).to(device)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor).to(device)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor).to(device)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
    target_ntype_nodes = torch.cat((train_node, valid_node, test_node)).tolist()
    
    graph = dgl.DGLGraph(sum(edges))

    num_classes = torch.max(train_target).item()+1
    args.n_classes = num_classes
    

    
    test_macro_f1 = 0
    test_micro_f1 = 0
    test_macro_precision = 0
    test_micro_precision = 0
    test_macro_recall = 0
    test_micro_recall = 0
    results = {'Macro_F1':[], 'Micro_F1':[], 'Train_time':[]}
    
    graph, graph_homos = preprocess_graph(graph, target_ntype_nodes, args.threshold, args.K, args.sample_times)
    print("--- processing data takes : %s seconds ---" % (time.time() - start_time_process))

    graph = graph.to(device)

    graph_homos = [graph.to(device) for graph in graph_homos]
    total_starttime = datetime.datetime.now()
    for seed in [11, 88, 66, 72, 21]:

        set_seed(seed)
        
        features = torch.from_numpy(node_features).type(torch.FloatTensor)
        args.n_nodes, args.feat_dim = features.shape
        input = features.to(device), graph, graph_homos
        
        model = SHAN(args).to(device)
        
        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                        weight_decay=args.weight_decay)
        loss_fuc = torch.nn.CrossEntropyLoss()
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_loss = 10000
        best_val_f1 = 0
        best_test_loss = 0
        best_test_f1 = 0
        best_test_micro_f1 = 0
        counter = 0

        starttime = datetime.datetime.now()
        train_times = []
        for i in tqdm(range(args.epochs), total=args.epochs):
            # logging.info('Epoch: {}'.format(i+1))  
            model.zero_grad()
            model.train()
            predict = model(input)
            train_predict = predict[train_node]
            train_loss = loss_fuc(train_predict, train_target)
            
            train_loss.backward()
            optimizer.step()
            
            train_f1 = f1_score(train_target.cpu(), torch.argmax(train_predict.detach(),dim=1).cpu(), average='macro') * 100
            
            # Valid
            model.eval()
            with torch.no_grad():

                predict = model(input)
                val_predict = predict[valid_node]
                val_loss = loss_fuc(val_predict, valid_target)
                val_f1 = f1_score(valid_target.cpu(), torch.argmax(val_predict.detach(),dim=1).cpu(), average='macro') * 100
                
                if (i+1) % 50 == 0 :
                    logging.info('Epoch: {}'.format(i+1))  
                    logging.info('Train - Loss: {:.2f}, Macro_F1: {:.2f}'.format(train_loss.detach().cpu(), train_f1 ))
                    logging.info('Valid - Loss: {:.2f}, Macro_F1: {:.2f}'.format(val_loss.detach().cpu(), val_f1))
 
                test_predict = predict[test_node]
                test_loss = loss_fuc(test_predict, test_target)
                
                test_f1 = f1_score(test_target.cpu(), torch.argmax(test_predict.detach(),dim=1).cpu(), average='macro') * 100
                test_micro_f1 = f1_score(test_target.cpu(), torch.argmax(test_predict.detach(),dim=1).cpu(), average='micro') * 100
            
            
            if torch.isnan(train_loss.detach().cpu()):
                logging.info('---------------nan occured, Train Break--------------------')
                break            
            
            # if val_f1  > best_val_f1 :
            if val_loss.item() < best_val_loss:
                counter = 0
                best_train_loss = train_loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_loss = val_loss.detach().cpu().numpy()
                best_val_f1 = val_f1
                best_test_loss = test_loss
                best_test_f1 = test_f1
                best_test_micro_f1 = test_micro_f1

            else:
                counter +=1
                if counter > args.patience and (i + 1) > args.min_epochs:
                    
                    logging.info('-----Train Finish(early stop)-----')
                    break
        
        endtime = datetime.datetime.now()
        train_time = (endtime - starttime).seconds/(i + 1)

        logging.info('-----Best Results-----')
        logging.info('Train - Loss: {:.2f}, Macro_F1: {:.2f}'.format(best_train_loss, best_train_f1))
        logging.info('Valid - Loss: {:.2f}, Macro_F1: {:.2f}'.format(best_val_loss, best_val_f1))
        logging.info('Test - Loss: {:.2f}, Macro_F1: {:.2f}, Micro_F1: {:.2f}'.format(test_loss, best_test_f1, best_test_micro_f1))
        logging.info('平均训练时间：{}S'.format(train_time))
        logging.info('\n')
        results['Macro_F1'].append(best_test_f1)
        results['Micro_F1'].append(best_test_micro_f1)
        results['Train_time'].append(train_time)
        
    total_endtime = datetime.datetime.now()
    total_train_time = (total_endtime - total_starttime).seconds/5
    logging.info('总平均训练时间：{}S'.format(total_train_time))
    mean_macro_f1 = np.mean(results['Macro_F1'])
    std_macro_f1 = np.std(results['Macro_F1'])
    mean_micro_f1 = np.mean(results['Micro_F1'])
    std_micro_f1 = np.std(results['Micro_F1'])
    mean_train_time = np.mean(results['Train_time'])
    logging.info('results: '+ str(results))
    logging.info('mean_macro_f1: {:.2f}\,±\,{:.2f}, mean_micro_f1: {:.2f}\,±\,{:.2f}, per_epoch_train_time: {}S\n'.format(mean_macro_f1, std_macro_f1, mean_micro_f1, std_micro_f1,  mean_train_time))


