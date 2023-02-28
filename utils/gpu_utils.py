# coding=utf-8

import warnings

from collections import OrderedDict

import pprint

try:
    import pynvml  # nvidia-ml provides utility for NVIDIA management

    HAS_NVML = True
except:
    HAS_NVML = False

import sys
import time


def auto_select_gpu(threshold=1000, show_info=True, num_gpu = 1):
    '''
    Select gpu which has largest free memory
    :param threshold: MB
    :param show_info:
    :return:
    '''

    def KB2MB(memory):
        return memory / 1024.0 / 1024.0

    if HAS_NVML:
        print('=====================Auto Select GPU=================================')
        valid_gpu = {}
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device_name = pynvml.nvmlDeviceGetName(handle)
            fanspeed = pynvml.nvmlDeviceGetFanSpeed(handle)
            powerusage = pynvml.nvmlDeviceGetPowerUsage(handle) / float(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

            if KB2MB(info.free) > threshold:
                valid_gpu[i] = OrderedDict({'Device':device_name, 'Free Memo':KB2MB(info.free), 'Fanspeed':fanspeed,
                                'Powerusage':powerusage,
                                'GPU_util':gpu_utilization,
                                            'Memory': info.used / float(info.total)
                                })

        pynvml.nvmlShutdown()
        if len(valid_gpu.keys()) == 0:
            print('=====================No valid GPU can use !=============================')
            return ''
        else:
            best_gpu, best_flag, best_memo = '', sys.maxsize, 0
            results = {}
            for k, vs in valid_gpu.items():
                gpu_info = 'Device:%s\tFree Memo:%.2f, GPU_util:%d, Fanspeed:%d, Powerusage:%.2f'%(vs[
                                                                                                             'Device'], vs['Free Memo'], vs['Fanspeed'], vs['GPU_util'], vs['Powerusage'])
                print(gpu_info)
                current_flag = sum([vs['Fanspeed'] / 100.0, vs['Powerusage'], vs['GPU_util'] / 100.0,
                                    vs['Memory'] * 0.9] )
                # print([vs['Fanspeed'] / 100.0, vs['Powerusage'], vs['GPU_util'] / 100.0,
                #        vs['Memory']])
                results[current_flag] = [k, vs['Device'], vs['Free Memo']]
            if len(results.keys()) != 0:
                for k in sorted(results.keys())[:num_gpu]:
                    info = '==================Using GPU %s:%s with free memory %.2f MB ==============='%(results[k][0],
                                                                                                         results[k][1],
                                                                                                         results[k][2])

                    if show_info:
                        print(info)
            else:
                print('=====================No valid GPU can use !=============================')

            return ','.join([str(results[k][0]) for k in sorted(results.keys())[:num_gpu]])

    else:
        info = 'pynvml is not installed, automatically select gpu is disabled!'
        warnings.warn(info)
        sys.exit(0)


def inquire_gpu(interval=1.0, show_info=True):
    """
    no valid gpu hang out
    :param interval: minute
    :return:
    """
    index = auto_select_gpu()
    interval = int(interval * 60)
    while index == '':
        time.sleep(interval)
        index = auto_select_gpu()
        if show_info:
            print('Sleep', time.strftime('%Y-%m-%d_%H:%M:%S'))

    return index


if __name__ == '__main__':
    for i in range(1000):
        print('select', auto_select_gpu(3000, num_gpu=2))
        time.sleep(1)
        print(
            '=========================================================================================================')

        # break
    # pynvml.nvmlInit()
    # print(pynvml.nvmlSystemGetDriverVersion())
    # deviceCount = pynvml.nvmlDeviceGetCount()
    # for i in range(deviceCount):
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    #     print("Device", i, ":", pynvml.nvmlDeviceGetName(handle))
    #     # fanspeed
    #     print(pynvml.nvmlDeviceGetFanSpeed(handle))
    #     powerusage = pynvml.nvmlDeviceGetPowerUsage(handle)
    #     print(powerusage / 1000)
    #     utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    #     print(utilization.gpu)  # gpu利用率
    #
    #     print(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))
    #     break

    # from pynvml.smi import nvidia_smi
    # nvsmi = nvidia_smi.getInstance()
    # print(nvsmi.DeviceQuery('memory.free, memory.total'))