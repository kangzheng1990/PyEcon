# -*- coding: utf-8 -*-


from scipy import *


# Dictionary List manuplation ------------------------------------------------------

def dict_sum(lst_of_dict,com_keys=None):
    """
    return a summed dict of common keys in multiple dictionaries
    Each element of key must be able to numerically sum up
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
            
    newdict={}
    for key in com_keys:
        val=0
        for Dict in lst_of_dict:
            val=val+Dict[key]
        newdict[key]=val
        
    return newdict



def dict_sumprod(lst_of_dict,weight,com_keys=None):
    """
    return a sumproduct dict of common keys in multiple dictionaries
    Each element of key must be able to numerically sum and prod
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
        
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        val=0
        for i in range(Nlst):
            Dict=lst_of_dict[i]
            val=val+weight[i]*Dict[key]
        newdict[key]=val
        
    return newdict



def dict_mean(lst_of_dict,com_keys=None):
    """
    return a mean dict of common keys in multiple dictionaries
    Each element of key must be able to numerically mean
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
    
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        val=0
        for Dict in lst_of_dict:
            val=val+(1.0/Nlst)*Dict[key]
        newdict[key]=val
        
    return newdict


    
    

def dict_std(lst_of_dict,com_keys=None):
    """
    return a mean dict of common keys in multiple dictionaries
    Each element of key must be able to numerically mean
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
    
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        temp=[]
        for Dict in lst_of_dict:
            temp.append(Dict[key])
        newdict[key]=std(array(temp),axis=0)
    
    return newdict




def dict_median(lst_of_dict,com_keys=None):
    """
    return a mean dict of common keys in multiple dictionaries
    Each element of key must be able to numerically mean
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
    
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        temp=[]
        for Dict in lst_of_dict:
            temp.append(Dict[key])
        newdict[key]=median(array(temp),axis=0)
    
    return newdict

