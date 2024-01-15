from utils import *
import os.path as osp
import copy
use_cuda = torch.cuda.is_available()
print('use_cuda:', use_cuda)
device = torch.device('cuda:0' if use_cuda else 'cpu')


def train(train_data, train_label, test_data, test_label, model, batch_size, lr):
    # you can change here for specific data
    train_data = train_data
    train_label = train_label
    test_data = test_data
    test_label = test_label
    train_loader = get_dataloader(train_data, train_label, batch_size=batch_size)
    test_loader = get_dataloader(test_data, test_label, batch_size=batch_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn_train = LabelSmoothing()
    loss_fn_test = nn.CrossEntropyLoss()
    def save_model(name):
        save_path = './model_param'
        ensure_path(save_path)
        previous_model = osp.join(save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(save_path, '{}.pth'.format(name)))
    max_acc = 0.0
    best_model = None
    max_f1 = 0.0
    total_acc = []
    total_f1 = []
    total_loss = []

    timer = Timer()
    patient = 50
    counter = 0
    model.to(device)
    for epoch in range(1, 100):
        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn_train, optimizer=optimizer)
        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, train ,loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))
        loss_test, pred_test, act_test = predict(
            data_loader=test_loader, net=model, loss_fn=loss_fn_test
        )
        acc_test, f1_test, _ = get_metrics(y_pred=pred_test, y_true=act_test)
        print('epoch {}, test, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_test, acc_test, f1_test))
        if acc_test >= max_acc:
            max_acc = acc_test
            max_f1 = f1_test
            save_model('max_acc_' + str(round(max_acc, 4)))
            best_model = copy.deepcopy(model)
            print('New max ACC model saved, with the test ACC being:{}----f1:{}'.format(max_acc, max_f1))
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break


        print('epoch:{} ETA:{}/{} '.format(epoch,
                                           timer.measure(),
                                           timer.measure(epoch / 200)
                                           ))
    return total_acc, total_f1, total_loss, max_acc, max_f1, best_model


def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        out, link_loss, ent_loss = net(x_batch)

        loss1 = loss_fn(out, y_batch)
        loss = loss1 + 0.5 * link_loss + ent_loss
        #loss = loss1
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train
def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out,_,_ = net(x_batch)
            loss1 = loss_fn(out, y_batch)
            loss = loss1
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val



