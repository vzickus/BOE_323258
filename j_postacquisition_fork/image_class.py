# Simple class which serves as a container for image data and various bits of metadata
# (which are mostly derived from the associated plist file)
class ImageClass:
    def __init__(self):
        return
    def fi(self):
        return self.frameIndex
    def ph(self):
        return self.phase
    def ph_post(self):
        if ('postacquisition_phase' in self.plist):
            return self.plist['postacquisition_phase']
        else:
            return -1.0
    def ps_timestamp(self):
        if ('timestamp' in self.plist):
            return self.plist['timestamp']
        else:
            return -1.0            
    def ph_off(self):
        if ('phase_from_offline_sync_analysis' in self.plist):
            return self.plist['phase_from_offline_sync_analysis']
        else:
            return -1.0
    def z_step(self):
        if ('z_scan' in self.plist) and (self.plist['z_scan']['stepwise_definitely_stationary'] == True):
            return self.plist['z_scan']['stepwise_info']['stepwise_counter']
        else:
            return -1.0
    def time_r(self):
        if ('time_received' in self.plist):
            return self.plist['time_received']
        else:
            return -1.0
