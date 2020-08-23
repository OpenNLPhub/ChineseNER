# Archtecture of QWhaleNER

>Author: Waldeinsamkeit
>
>Description: Design the data structure of whole project , the implementation of every package



## Directory Structure

>- data
>  - static
>  - model
>  - result
>  - input
>- src
>  - models
>    - wrapper
>    - nn
>  - config
>    - nn
>    - training
>    - path
>  - data
>  - main.py
>  - metrics.py
>  - evaluate
>  - utils
>    - commonutil.py
>    - nn
>- README.md
>- LICENSE



**data.static**

Save pre-trained models which other people trained on large corpus. For example, bert-base-chinese



**data.model**

Save models'parameters or checkpoints which trained before. Using model's name to name the file.



**data.result**

Save the ouput of model's performance



**data.input**

the input data



**src.models.nn**

In this package,  all classes are inherited from torch.nn.Module. We try to pack the model as a layer. 

```python
class example_nn(nn.Module):
  def __init__(self)
    super(example_nn,self).__init__()
    
 	def forward(self):
    '''
    inherited from nn.Module. 
    Design the Calculation graph
    '''
    
 	def test(self,input):
    '''
    this method is called to get input and return output
    '''
  def cal_loss(self):
    '''
    for some models, we have to overwrite the loss function by ourselves.For exmaple conditional random field, we suggest to integrated special and complex loss function in model's class. 
    Of course, you can use common loss function from pytorch or create another class or new function.
    '''
  
  def _helper_function(self):
    '''
    if this function is specifictly related to logit of this model, we put add underline before the name of the function and put it there. Otherwise, we suggest to move this function under utils package.
    '''
```



**src.models.wrapper**

In this package, we wrapper models and packed them in classes which have same implementation. For every model, the wrapper class is inherited from a ModelWrapper class. In specific model class, we can also overwrite some method, like test,validate,train, which are unified implementation.

In addition, we try to write some sub model to assemble the ModelWrapper class. Like test,validate,train.

User can only call this class instead of classes under nn directory.

```python
class ModelWrapper(object):
  def __init__(self):
    pass
  def test(self):
    '''
    
    '''
    pass
 	def train(self):
    '''
    
    '''
    pass
  def validate(self):
    '''
    
    '''
    pass
  def train_step(self):
    '''
    '''
    pass


class  LSTM_CRF_Model(ModelWrapper):
  def __init__(self):
    pass
  @Overwrite()
  def train(self):
    pass
```



