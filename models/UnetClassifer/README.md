# Model Architecture

```
        Input                                                                         
          |                                                                           
          |                                                                           
       Backbone                                                                       
       | | | | |                                                                      
       | | | | feat5 -\                                                               
       | | | |        |                                                               
       | | | feat4 --up_concat                                                       
       | | |             |                                                            
       | | |            up4                                                         
       | | feat3 -\      |                                                            
       | |         \___up_concat                                                     
       | |               |                                                            
       | feat2 --\      up3                                                           
       |          \___up_concat                                                      
       |                |                                                             
       feat1 ---\      up2                                                           
                 \____up_concat----final                                             
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
                                                                                      
```