
# coding: utf-8

# # YanuX Cruncher #

# In[1]:

from model.jsonloader import JsonLoader
from model.wifilogstats import WifiLogStats


# In[2]:

path = 'data'
json_loader = JsonLoader(path)


# **# Data Samples**

# In[3]:

print(str(len(json_loader.json_data)))


# In[4]:

wifi_log_stats = WifiLogStats(json_loader.json_data)


# **# Unique MAC Addresses**

# In[5]:

print(len(wifi_log_stats.mac_addresses))


# In[6]:

print(wifi_log_stats.locations)

