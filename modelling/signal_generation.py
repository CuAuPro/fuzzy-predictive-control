import numpy as np
def drbsgen(fs, fb, prbs_length, seed=42):
    #   DRBSGEN Random binary signal generator.
    #   DRBSGEN generates signal with:
    #   fs - sampling frequency
    #   fb - bandwith of the DRBS signal
    #   prbs_length - length of the signal in seconds
    #   seed - set seed for random number generator (in order to exactly recreate results) 
    np.random.seed(seed)
    f_prbs = 3*fb
    N = int(np.around(fs/f_prbs, decimals=0))
    Ns = int(np.ceil((prbs_length*fs)/N))
    lb = int(np.ceil(prbs_length*fs))
    prbs = np.ones(int(lb))#*np.nan;
    
    for idx in range(1,Ns):
        x = np.around(np.random.uniform(0,1),decimals=0)
        if(x==0):
            x = 0
        prbs[((idx-1)*N+1):idx*N+1] = x
   
    t = np.arange(0,len(prbs))/fs
    t = t[0:lb]
    prbs = np.append(prbs[1:lb],prbs[-1])
    return prbs,t


def aprbsgen(fs, fb, prbs_length, min_val=0, max_val=1, seed=42):
    #   APRBS Amplitude Pseudo Random Binary Signal Generator.
    #   APRBS generates signal with:
    #   fs - sampling frequency
    #   fb - bandwith of the DRBS signal
    #   prbs_length - length of the signal in seconds
    #   seed - set seed for random number generator (in order to exactly recreate results) 
    np.random.seed(seed)
    prbs,t = drbsgen(fs, fb, prbs_length, seed)
    d = np.diff(prbs)
    idx = np.nonzero(d)[0]
    idx = np.concatenate(([1], idx))
    for ii in range(len(idx) - 1):
        amp = np.random.randn()
        prbs[idx[ii]:idx[ii+1]] = amp * prbs[idx[ii]]

    aprbs = normalize(prbs, 'range')
    aprbs = scale_range(aprbs, min_val, max_val)
    
    return aprbs, t
    
    
def normalize(data, method):
    if method == 'range':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    # Add other normalization methods here if needed
    else:
        raise ValueError("Invalid normalization method")
    
def scale_range(value, min_x, max_x):
    return (value * (max_x - min_x)) + min_x


def generate_reference_signal(ref_vals, N):
  """Generates a reference signal."""
  ref = np.zeros(shape=(N*ref_vals.shape[0]))
  for idx, ref_val in enumerate(ref_vals):
    ref[idx * N:(idx + 1) * N] = ref_val * np.ones(N)
  return ref