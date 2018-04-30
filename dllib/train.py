from .imports import *
from .utils import to_np
#from .model import *
#from .metrics import *
from .lr_sched import *
from .logger import Logger
#import importlib
tqdm.monitor_interval = 0

class Trainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion='default', pre_trained=True, name ='default', 
                                                                                                        metrics_calc='accuracy'):
        self.model = model
        #self.datasets = datasets
        #self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.pre_trained = pre_trained
        self.name = name
        self.best_acc = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
        #self.metrics = metrics_calc
        #self.metrics_module = importlib.import_module('..metrics', __name__)
        #self.acc_functions = {'f2': metrics.f2,'accuracy': metrics.accuracy}
        
        #self.monitor = boilerplate.VisdomMonitor(port=80)
 
        
    def train_model(self, optimizer=None, scheduler=None, num_epochs=10, metrics=None):
        since = time.time()
        if not (optimizer is None):
            self.optimizer = optimizer
            self.best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
        if not (metrics is None):
            self.metrics = metrics    
        self.history = {}
        iteration = 0
        #best_model_wts = copy.deepcopy(self.model.state_dict())
        #best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
        
        #dataset_sizes = {x: len(self.datasets[x]) for x in ['train', 'val']}
        #class_names = self.datasets['train'].classes
        dataloaders = {'train':self.train_loader,'val':self.val_loader}

        #save training process to simple Logger function
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        logger = Logger(os.path.join('checkpoint', 'log.txt'), title=self.name)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Check if scheduler is present, if so then check if it has apply_batch attribute.  
                    # if scheduler is present and apply_batch is false, then scheduler step
                    # if scheduler is present and doesn't have a apply_batch attribute, then scheduler step
                    if not (scheduler is None):
                        if hasattr(scheduler, 'apply_batch'):
                            if not scheduler.apply_batch:
                                scheduler.step()
                        else: scheduler.step()

                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                dataset_sizes = 0
                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    # get the inputs
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    if phase == 'train':
                        if hasattr(scheduler, 'apply_batch'):
                            if scheduler.apply_batch:
                                scheduler.step()
                        outputs, loss = self.fit_on_batch(inputs,labels,self.criterion,self.optimizer)
                        iteration += 1
                        self.history.setdefault('lr', []).append(self.optimizer.param_groups[0]['lr'])
                        self.history.setdefault('loss', []).append(loss.item())
                        self.history.setdefault('iterations', []).append(iteration)
                    else:
                        outputs, loss = self.evaluate_on_batch(inputs,labels,self.criterion)
                                        
                    #_, preds = torch.max(outputs.data, 1)
                    # Run the function according the metrics specified for scoring the model
                    #score = getattr(self.metrics_module, self.metrics)(outputs.cpu().data.numpy(),labels.cpu().numpy())
                    score = self.model.calculate_metrics(outputs,labels)                        
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    #running_corrects += torch.sum(preds == make_var(labels).data)
                    running_corrects += score * inputs.size(0)
                    dataset_sizes += inputs.size(0)
                   
                epoch_loss = running_loss / dataset_sizes
                epoch_acc = running_corrects / dataset_sizes

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                # deep copy the model
                # remember best accuracy and save checkpoint
                if phase == 'val':
                    is_best = epoch_acc > self.best_acc
                    valid_loss = epoch_loss
                    valid_acc = epoch_acc
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best_acc': self.best_acc,
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best)
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
                        self.best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
                else:
                    train_loss = epoch_loss
                    train_acc = epoch_acc        
            # append logger file
            logger.append([self.optimizer.param_groups[0]['lr'], train_loss, valid_loss, train_acc, valid_acc])
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))
        
        logger.close()
        logger.plot()
 
        # load best model weights and optimizer parameters
        self.model.load_state_dict(self.best_model_wts)
        self.optimizer.load_state_dict(self.best_optimizer_params)
        #return self.model
     
    def fit_on_batch(self, x, y_true, loss_fn, optimizer):
        """Trains the model on a single batch of examples.
        This is a training function for a basic classifier. For more complex models,
        you should write your own training function.
        NOTE: Before you call this, make sure to do `model.train(True)`.
        Parameters
        ----------
        model: nn.Module
            The model to train.
        x: Tensor 
            Image tensors should have size (batch_size, in_channels, height, width).
        y: Tensor
            Contains the label indices (not one-hot encoded).
        loss_fn: 
            The loss function to use.
        optimizer: 
            The SGD optimizer to use.
        Returns
        -------
        dict
            The computed metrics for this batch.
        """
        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = self.model(x)

        # Compute loss
        #y_true = make_var(y, dtype=np.int)
        #loss = loss_fn(outputs, y_true)
        loss = self.model.calculate_loss(outputs,y_true)

        # Backward pass
        loss.backward()
        optimizer.step()

        return outputs, loss
    
    def evaluate_on_batch(self, x, y_true, loss_fn, metrics=["loss", "acc"]):
        """Evaluates the model on a single batch of examples.

        This is a evaluation function for a basic classifier. For more complex models,
        you should write your own evaluation function.    

        NOTE: Before you call this, make sure to do `model.train(False)`.

        Parameters
        ----------
        model: nn.Module
            Needed to make the predictions.
        x: Tensor or numpy array 
            Image tensors should have size (batch_size, in_channels, height, width).
        y: Tensor or numpy array 
            Contains the label indices (not one-hot encoded)
        loss_fn: optional
            The loss function used to compute the loss. Required when the
            metrics include "loss".
        metrics: list
            Which metrics to compute over the batch.

        Returns
        -------
        dict
            The computed metrics for this batch.
        """
        self.model.train(False)
        outputs = self.model(x)
        #y_true = make_var(y, dtype=np.int, volatile=True)
        #loss = loss_fn(outputs, y_true)
        loss = self.model.calculate_loss(outputs,y_true)
        return outputs, loss
    
    def lr_find(self, start_lr=1e-5, end_lr=10, steps=None):
        """Finds the optimal learning rate for training.
        Typically you'd do this on a model that has not been trained yet.
        However, calling find_lr() on a (partially) trained model is OK too;
        the state of the model and optimizer are preserved so that find_lr()
        won't actually change the model's parameters.
        Parameters
        ----------
        start_lr: float (optional)
            The learning rate to start with (should be quite small).
        end_lr: float (optional)
            The maximum learning rate to try (should be large-ish).
        steps: int (optional)
            How many batches to evaluate, at most. If not specified,
            find_lr() runs for a single epoch. As a rule of thumb, 100
            steps seems to work well.
        """
        self._save_state()

        one_epoch = len(self.train_loader)
        epochs = 1
        if steps is None:
            steps = one_epoch
        elif one_epoch < steps:
            epochs = (steps + one_epoch - 1) // one_epoch
       
        #lr_history = []
        #loss_history = []
        self.history = {}
        iteration = 0
        best_loss = 1e9
        prior_loss = 1.0
        
        # Set initial Learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
        lr_decay = 10**((np.log10(end_lr)-np.log10(start_lr))/float(steps))
        lr_value = start_lr
        should_stop = False
        
        
        print("Trying learning rates between %g and %g over %d steps (%d epochs)" %
              (start_lr, end_lr, steps, epochs))
        
        for epoch in range(epochs):
            if should_stop:
                break
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
                        
            self.model.train(True)  # Set model to training mode
                   
            # Iterate over data.
            #for batch_idx, data in tqdm(enumerate(self.train_loader)):
            for inputs, labels in tqdm(self.train_loader):
                # get the inputs
                

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                iteration += 1
                
                outputs, loss = self.fit_on_batch(inputs,labels,self.criterion,self.optimizer)
                
                #score = getattr(self.metrics_module, self.metrics)(outputs.cpu().data.numpy(),labels.cpu().numpy())
                #score = self.model.calculate_metrics(outputs,make_var(labels, dtype=np.int))  
                score = self.model.calculate_metrics(outputs,labels) 
                
                #_, preds = torch.max(outputs.data, 1)
                #corrects = torch.sum(preds == make_var(labels).data)

                # statistics
                self.history.setdefault('lr', []).append(lr_value)
                self.history.setdefault('loss', []).append(loss.item())
                self.history.setdefault('iterations', []).append(iteration)
                #loss_history.append(loss.data[0])
                #lr_history.append(lr_value)
                                
                print('Iteration No:{} Learn rate {:.2E} Batch Loss: {:.4f} Batch Accuracy: {:.4f} '.format(
                    iteration, lr_value, loss.item(), score))
                
                if loss.item() < best_loss:
                    best_loss = loss.item()

                if math.isnan(loss.item()) or (loss.item() > prior_loss*20 and prior_loss > .01) or iteration >= steps - 1:
                    should_stop = True
                    break
                #set prior_iteration loss to current loss
                prior_loss = loss.item()  
                # Increase the learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                lr_value *= lr_decay            
         
        self.loss_lr_plot()         
        self._restore_state()
     
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')


    def loss_lr_plot(self,figsize=(12, 6)):
        fig = plt.figure(figsize=figsize)
        plt.ylabel("loss", fontsize=16)
        plt.xlabel("learning rate (log scale)", fontsize=16)
        plt.xscale("log")
        plt.plot(self.history['lr'], self.history['loss'])
        plt.show()
        
    def _save_state(self):
        state = {}
        state["model"] = copy.deepcopy(self.model.state_dict())
        state["optimizer"] = copy.deepcopy(self.optimizer.state_dict())
        self._saved_state = state

    def _restore_state(self):
        state = self._saved_state
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
    
    def save_model(self, m, p): torch.save(m.state_dict(), p)

    def load_model(self, m, p): m.load_state_dict(torch.load(p, map_location=lambda storage, loc: storage)) 

    def predict(self, is_test=False):
        dl = self.val_loader
        preda,_ = self.predict_with_targs_(self.model, dl)
        return to_np(torch.cat(preda))
        
    def predict_with_targs(self, is_test=False):
        preda,targa = self.predict_with_targs_(self.model, self.val_loader)
        return to_np(torch.cat(preda)), to_np(torch.cat(targa))
 
    def predict_with_targs_(self, m, dl):
        self.model.eval()
        #if hasattr(m, 'reset'): m.reset()
        res = []
        for x,y in tqdm(iter(self.val_loader)): res.append([F.softmax(self.model(x.to(self.device)),dim=1),y])
        return zip(*res)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, 'checkpoint/'+self.name+'_'+filename)
        if is_best:
            shutil.copyfile('checkpoint/'+self.name+'_'+filename, 'model_best.pth.tar')    
    


