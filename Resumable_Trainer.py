import pandas as pd 
import numpy as np

'''
simplified history structure of the Keras.engine.History.
only 'epoch' and 'history' parts are stored for possible 
visualization usage.
'''
class History_data:
    def __init__(self,epoch=[],history={}):
        self.epoch=epoch
        self.history=history
    def toDataFrame(self):
        df = pd.DataFrame(self.history)
        df.index = self.epoch
        return df
        
''' #Resumable Trainer with callbacks#
!!! The callbacks used should be 'picklable' !!!

-> A simple wrapper of current Keras.Model
-> Try to solve the problem that Keras does not store the
    states of callbacks through model.save, which really annoys
    people who have to resume training model frequently.
    
-> The trainer will train model from '@start_epoch'+1 to 
    final '@epochs', while stop every '@ep_turn'. People
    can use 'Pickle' outside to dump and load this trainer
    with trainning state automatically saved.
-> To start one turn of training, simply called trainer.train
    with user-maintained model, place to store the model, and
    the parameter for Model.fit except 'initial_epoch', 'epochs'
    , and 'callbacks'.  A simplified history instance is return
    similar to original Model.fit.
    [?] => The reason I leave model out for user to maintain is 
        to increase the flexibaility for user to change their
        models. A 'set_callbacks' function is provided for a
        similar reason.
-> A 'isStopped'function is offered to check stop.

-> If one want to start a new Resumable trainner without lossing 
    history, just init the new trainer with appropriate start_epoch
    , history = oldtrainer.history, and new settings.

'''        
class ResumableTrainer_callback:
    def __init__(self,epochs,ep_turn,start_epoch = 0,earlystop=False,callbacks=None,
                 history = History_data()):
        self.epochs = epochs
        self.ep_turn = ep_turn
        self.start_epoch = start_epoch
        self.history = history
        self.callbacks = callbacks
        self.earlystop = earlystop
        self.turn = -1
        self.stopped = False
        
    def isStopped(self):
        return self.stopped
        
    def set_callbacks(self,callbacks,extend=True):
        if extend: self.callbacks.extend(callbacks)
        else: self.callbacks = callbacks
        
    def new_turn(self):
        self.turn += 1
        self.start = self.start_epoch + self.turn*self.ep_turn
        self.end = min(self.epochs, self.start+self.ep_turn)
    
    def train(self,model,filename,*para,**kpara):
        self.new_turn()
        hist = model.fit(*para,**kpara,initial_epoch=self.start,epochs=self.end,callbacks = self.callbacks)
        if self.end==self.epochs or self.earlystop and model.stop_training:
            self.stopped = True
        else: model.save(filename, overwrite=True)   
        self.clear_callbacks()
            
        return self.history_comb(hist)
    
    def clear_callbacks(self):
        keys = ['validation_data','model']
        for i in self.callbacks:
            for j in keys:
                if i[j]: i[j] = None
    
    def history_comb(self,hist):
        self.history.epoch.extend(list(range(self.start+1,self.end+1)))
        for k, v in hist.history.items():
            self.history.history.setdefault(k, []).extend(v)
        return self.history