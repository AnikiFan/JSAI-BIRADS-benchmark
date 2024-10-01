class DummyScheduler:
    def __init__(self,*params,**kwargs):
        """
        不执行任何操作的schedular
        :param kwargs:
        """
        pass
    def step(self,*params,**kwargs):
        pass

    def state_dict(self,*params,**kwargs):
        return "this is a dummy schedular!"
