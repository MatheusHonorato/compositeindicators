from .means import weighted_arithimetic_mean

def pair_iterator(direction_map):
    """
    Generates pairs of indices from the direction_map for optimization.
    Parameters:
    direction_map : list
        A list of indices representing the directions.
    Returns: generator
        A generator that yields pairs of indices from the direction_map.
    """
    while True:
        for n in ((direction_map[i],direction_map[j]) for j in range(len(direction_map)) for i in range(j)):
            yield n
            
def optimization(direction_map=None, dstep= 0.01,
                 limit_dquality = 0.0001, limit_nstep = 5000,
                 upper_limits=None, lower_limits=None,
                 mu_x = weighted_arithimetic_mean
                ):

    direction_map_=direction_map
    upper_limits_=upper_limits
    lower_limits_=lower_limits
    dstep_=dstep
    limit_dquality_=limit_dquality
    limit_nstep_=limit_nstep
    mu_x_=mu_x
    
    def Optimizer(normal_data, quality_indicator, coef=None):
        upper_limits = upper_limits_
        lower_limits = lower_limits_
        direction_map = direction_map_
        dstep=dstep_
        limit_dquality=limit_dquality_
        limit_nstep=limit_nstep_
        mu_x=mu_x_
        if upper_limits is None:
            upper_limits = [1 for _ in range(normal_data.shape[1])]
        if lower_limits_ is None:
            lower_limits = [0 for _ in range(normal_data.shape[1])]
        if direction_map_ is None:
            direction_map = [i for i in range(normal_data.shape[1])]
            
        pairs_gen = pair_iterator(direction_map)
        n_directions = (len(direction_map) * (len(direction_map)-1)) / 2
        
        pair = next(pairs_gen)
        covered_directions = 0
        n_step = 0
        
        if coef is None:
            coef = [1/normal_data.shape[1] for _ in range(normal_data.shape[1])]
        ci = normal_data.apply(lambda row: mu_x(coef, row), axis=1)
        quality = quality_indicator(ci)
        old_quality = 2 * limit_dquality + quality
        
        while n_step < limit_nstep and covered_directions < n_directions:
            updated = False
            actual_dstep=dstep
            if coef[pair[0]] + actual_dstep > upper_limits[pair[0]]:
                actual_dstep = upper_limits[pair[0]] - coef[pair[0]]
            if coef[pair[1]] - actual_dstep < lower_limits[pair[1]]:
                actual_dstep = coef[pair[1]] - lower_limits[pair[1]]
            # print(n_step, pair)
            # print(quality)
            # print(covered_directions, n_directions)
            
            if actual_dstep>0:
                new_coef = coef.copy()
                new_coef[pair[0]] += actual_dstep
                new_coef[pair[1]] -= actual_dstep
                
                ci = normal_data.apply(lambda row: mu_x(new_coef, row), axis=1)
                new_quality = quality_indicator(ci)
                if new_quality > quality:
                    quality = new_quality
                    coef = new_coef
                    updated = True
    
            # print('coef')
            # print(coef)
            actual_dstep=dstep
            if coef[pair[1]] + actual_dstep > upper_limits[pair[1]]:
                actual_dstep = upper_limits[pair[1]] - coef[pair[1]]
            if coef[pair[0]] - actual_dstep < lower_limits[pair[0]]:
                actual_dstep = coef[pair[0]] - lower_limits[pair[0]]
                
            if not updated and actual_dstep>0:
                new_coef = coef.copy()
                new_coef[pair[0]] -= actual_dstep
                new_coef[pair[1]] += actual_dstep
                
                ci = normal_data.apply(lambda row: mu_x(new_coef, row), axis=1)
                new_quality = quality_indicator(ci)
                if new_quality > quality:
                    quality = new_quality
                    coef = new_coef
                    
            if limit_dquality <= abs(quality - old_quality):
                covered_directions = 0
            else:
                covered_directions += 1
        
            old_quality = quality
            pair = next(pairs_gen)
            n_step += 1
            
        return coef
    return Optimizer
