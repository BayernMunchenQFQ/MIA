# -- coding: utf-8 --

import argparse

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--train_data_path', type=str, default='./dbpedia/train.csv',
                    metavar='N', help='this is the path of training data')
parser.add_argument('--val_data_path', type=str, default='./dbpedia/val.csv',
                    metavar='N', help='this is the path of val data')
parser.add_argument('--test_data_path', type=str, default='./dbpedia/test.csv',
                    metavar='N', help='this is the path of test data')

parser.add_argument('--shadow_train_data_path', type=str, default='./dbpedia/train.csv',
                    metavar='N', help='this is the path of training data')
parser.add_argument('--shadow_val_data_path', type=str, default='./dbpedia/val.csv',
                    metavar='N', help='this is the path of val data')
parser.add_argument('--shadow_test_data_path', type=str, default='./dbpedia/test.csv',
                    metavar='N', help='this is the path of test data')

parser.add_argument('--target_train_num', type=int, default=10000,
                    metavar='N', help='seed')
parser.add_argument('--shadow_train_num', type=int, default=10000,
                    metavar='N', help='seed')
parser.add_argument('--attack_train_num', type=int, default=10000,
                    metavar='N', help='seed')
parser.add_argument('--attack_val_num', type=int, default=10000,
                    metavar='N', help='seed')
parser.add_argument('--attack_test_num', type=int, default=10000,
                    metavar='N', help='seed')
parser.add_argument('--LM_train_num', type=int, default=10000,
                    metavar='N', help='seed')

parser.add_argument('--topk', type=int, default=10,
                    metavar='N', help='cliptopk')
parser.add_argument('--num_trees', type=int, default=10,
                    metavar='N', help='the number of the trees for randomforest')
parser.add_argument('--test_size', type=int, default=0.5,
                    metavar='N', help='影子模型所用数据的比例')
parser.add_argument('--C', type=float, default=5,
                    metavar='N', help='SVC惩罚参数')
parser.add_argument('--shadow_C', type=float, default=10,
                    metavar='N', help='SVC惩罚参数')
parser.add_argument('--shuffle', type=bool, default=True,
                    metavar='N', help='是否将数据打乱')



parser.add_argument('--max_length', type=int, default=300,
                    metavar='N', help='this is the max length of sentences')
parser.add_argument('--target_model_name', type=str, default='textcnn',
                    metavar='N', help='select cnn,rnn')
parser.add_argument('--target_model_path', type=str, default='./models/newsgroups/textcnn_target_model.pkl',
                    metavar='N', help='select cnn,rnn')
parser.add_argument('--shadow_model_name', type=str, default='textcnn',
                    metavar='N', help='select cnn,rnn')
parser.add_argument('--shadow_model_path', type=str, default='./models/newsgroups/testcnn_',
                    metavar='N', help='select cnn,rnn')
parser.add_argument('--shadow_model_num', type=int, default=15,
                    metavar='N', help='seed')
parser.add_argument('--attack_model_name', type=str, default='textcnn',
                    metavar='N', help='select cnn,rnn')
parser.add_argument('--attack_model_path', type=str, default='./models/newsgroups/tree_attack_model.pkl',
                    metavar='N', help='select cnn,rnn')
parser.add_argument('--num_class', type=int, default=20,
                    metavar='N', help='this is the number of data class')
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='N', help='this is the learning rate in train')
parser.add_argument('--batch_size', type=int, default=128,
                    metavar='N', help='this is the batch size for data')
parser.add_argument('--epoch', type=int, default=30,
                    metavar='N', help='this is the times for training')

parser.add_argument('--vocab_size', type=int, default=80000,
                    metavar='N', help='vocab size')
parser.add_argument('--seed', type=int, default=1,
                    metavar='N', help='seed')
parser.add_argument('--gpu_id', type=int, default=0,
                    metavar='N', help='seed')

config = parser.parse_args()
