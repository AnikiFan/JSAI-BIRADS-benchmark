class EarlyStopping:
    def __init__(self, patience:int=5, min_delta:float=0.001,min_train_loss:float=float('inf')):
        """
        初始化 EarlyStopping 实例。

        :param patience: 在多少个 epoch 内验证损失没有改善就停止训练。
        :param min_delta: 判定损失改善的最小变化幅度。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_train_loss = min_train_loss
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self,train_loss:float,val_loss:float)->None:
        # 如果验证损失有显著改善，重置计数器和最佳损失
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif train_loss < self.min_train_loss and val_loss < self.min_val_loss:
            # 如果验证损失没有改善，计数器加1
            self.counter += 1
            # 如果计数器达到耐心值，设置早停标志
            if self.counter >= self.patience:
                self.early_stop = True