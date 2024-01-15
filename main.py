from MCGP import *
from train_model import *
import h5py
from Tee import *


def save_model(name):
    save_path = './model_param'
    ensure_path(save_path)
    previous_model = osp.join(save_path, '{}.pth'.format(name))
    if os.path.exists(previous_model):
        os.remove(previous_model)
    torch.save(model.state_dict(), osp.join(save_path, '{}.pth'.format(name)))


if __name__ == '__main__':

    print('Train：session1  Test：session3')
    log_path = './log'
    ensure_path(log_path)
    log_path = os.path.join(log_path, 'session1---session3.txt')
    sys.stdout = Tee(log_path)
    seed_all(2023)
    set_gpu('0')
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # load data
    file_path_train = '/home/lhg/processed_data/SEED_DE/mixture/overlap0_step_10'
    file_path_test = '/home/lhg/processed_data/SEED_DE/mixture/overlap0_step_10'
    train_data_name = 'session1.hdf'
    train_data_path = os.path.join(file_path_train, train_data_name)
    dataset = h5py.File(train_data_path, 'r')
    train_data = np.array(dataset['data'])
    train_label = np.array(dataset['label'])
    train_data = np.concatenate(train_data, axis=0)
    train_data = np.expand_dims(train_data, axis=1)
    train_label = np.concatenate(train_label, axis=0)
    print('>>> total -->Train Data:{}Train Label:{}'.format(train_data.shape, train_label.shape))
    test_data_name = 'session3.hdf'
    test_data_path = os.path.join(file_path_train, test_data_name)
    dataset = h5py.File(test_data_path, 'r')
    test_data = np.array(dataset['data'])
    test_label = np.array(dataset['label'])
    test_data = np.concatenate(test_data, axis=0)
    test_data = np.expand_dims(test_data, axis=1)
    test_label = np.concatenate(test_label, axis=0)
    print('>>> total -->Test Data:{} Test Label:{}'.format(test_data.shape, test_label.shape))
    train_data, test_data = normalize_v2(train_data, test_data)
    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).long()
    print('Data and label prepared!')
    print('>>> Test Data:{} Test Label:{}'.format(test_data.shape, test_label.shape))
    print('>>> Train Data:{}Train Label:{}'.format(train_data.shape, train_label.shape))
    print('----------------------')
    input_size = (1, 310, 10)
    model = MCGPnet(num_class=15, input_size=input_size, sampling_rate=10, num_T=64,
                    out_graph=30, out_graph_=10, dropout_rate=0.25, pool=4,
                    pool_step_rate=0.25)
    model.to(device)
    va_val = Averager()
    vf_val = Averager()
    preds, acts = [], []
    total_acc, total_f1, total_loss, max_acc, max_f1, max_model = train(
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        test_label=test_label,
        model=model,
        batch_size=64,
        lr=0.0015)
    print('total_acc:', total_acc, 'total_f1:', total_f1)

