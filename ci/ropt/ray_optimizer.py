from .means import weighted_arithimetic_mean
from random import random, randint

def rand_pair_iterator(direction_map):
    """
    Generates pairs of indices from the direction_map for optimization.
    Parameters:
    direction_map : list
        A list of indices representing the directions.
    Returns: generator
        A generator that yields pairs of indices from the direction_map.
    """
    while True:
        dmx = direction_map.copy()
        while len(dmx)>0:
            choice = randint(0,len(dmx)-1)
            dmx[choice], dmx[-1] = dmx[-1], dmx[choice]
            choice_x = dmx.pop()
            
            dmy = direction_map.copy()
            while len(dmy)>0:
                choice = randint(0,len(dmy)-1)
                dmy[choice], dmy[-1] = dmy[-1], dmy[choice]
                choice_y = dmy.pop()

                if random()>0.5:
                    yield (choice_x, choice_y)
                else:
                    yield (choice_y, choice_x)
                # for n in ((direction_map[i],direction_map[j]) for j in range(len(direction_map)) for i in range(j)):
                #     yield n

def search_step(
        coef, normal_data, mu_x, quality_indicator, 
        quality, reach, i, j, depth=10, search=3, i_search=0
    ):
    new_coef = coef.copy()
    new_coef[i] += reach
    new_coef[j] -= reach
    ci = normal_data.apply(lambda row: mu_x(new_coef, row), axis=1)
    new_quality = quality_indicator(ci)
    
    if depth != 0:
        if i_search == 0:
            if new_quality <= quality:
                # print((i,j,reach,depth,i_search,search,"worse"))
                reach = search_step(coef, normal_data, mu_x, quality_indicator, new_quality, reach/2, i, j, depth-1, search)
            else:
                # print((i,j,reach,depth,i_search,search,"better"))
                _reach = search_step(coef, normal_data, mu_x, quality_indicator, new_quality, reach/2, i, j, depth-1, search, i_search + 1)
                reach = _reach or reach
        elif i_search != search:
            if new_quality <= quality:
                # print((i,j,reach,depth,i_search,search,"worse"))
                reach = search_step(coef, normal_data, mu_x, quality_indicator, new_quality, reach*1.5, i, j, depth-1, search, i_search + 1)
            else:
                # print((i,j,reach,depth,i_search,search,"better"))
                _reach = search_step(coef, normal_data, mu_x, quality_indicator, new_quality, reach/2, i, j, depth-1, search, i_search + 1)
                reach = _reach or reach
        else:
            if new_quality <= quality:
                # print((i,j,reach,depth,i_search,search,"worse"))
                return None
            else:
                return reach
                # print((i,j,reach,depth,i_search,search,"better"))
            
    return reach
    

def ray_clock_optimization(direction_map=None, dstep= 0.01,
                 limit_dquality = 0.0001, limit_nstep = 5000,
                 upper_limits=None, lower_limits=None,
                 mu_x = weighted_arithimetic_mean,
                 ray_depth = 10, ray_search = 2
                ):

    direction_map_=direction_map
    upper_limits_=upper_limits
    lower_limits_=lower_limits
    dstep_=dstep
    limit_dquality_=limit_dquality
    limit_nstep_=limit_nstep
    mu_x_=mu_x
    ray_depth_=ray_depth
    ray_search_=ray_search
    
    def Optimizer(normal_data, quality_indicator, coef=None):
        upper_limits = upper_limits_
        lower_limits = lower_limits_
        direction_map = direction_map_
        dstep=dstep_
        limit_dquality=limit_dquality_
        limit_nstep=limit_nstep_
        mu_x=mu_x_
        ray_depth=ray_depth_
        ray_search=ray_search_
        if upper_limits is None:
            upper_limits = [1 for _ in range(normal_data.shape[1])]
        if lower_limits_ is None:
            lower_limits = [0 for _ in range(normal_data.shape[1])]
        if direction_map_ is None:
            direction_map = [i for i in range(normal_data.shape[1])]
            
        pairs_gen = rand_pair_iterator(direction_map)
        n_directions = (len(direction_map) * (len(direction_map)-1)) / 2
        
        pair = next(pairs_gen)
        covered_directions = 0
        n_step = 0
        
        if coef is None:
            coef = [1/normal_data.shape[1] for _ in range(normal_data.shape[1])]
        ci = normal_data.apply(lambda row: mu_x(coef, row), axis=1)
        quality = quality_indicator(ci)
        old_quality = 2 * limit_dquality + quality
        
        while n_step < limit_nstep and covered_directions < 3* n_directions:
            # print((n_step, limit_nstep, covered_directions, n_directions))
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
                # print(new_quality - quality)
                if new_quality > quality:
                    quality = new_quality
                    coef = new_coef
                    updated = True
                elif abs(new_quality - quality) < limit_dquality:
                    reach  = min(
                        upper_limits[pair[0]] - coef[pair[0]],
                        coef[pair[1]]-lower_limits[pair[1]]
                    )
                    # print(("here",reach))
                    if reach != 0:
                        actual_dstep = search_step(
                            coef, normal_data, mu_x, quality_indicator, 
                            quality, reach, pair[0], pair[1],
                            ray_depth, ray_search
                        )
                        if actual_dstep < dstep:
                            actual_dstep = dstep
                        if coef[pair[0]] + actual_dstep > upper_limits[pair[0]]:
                            actual_dstep = upper_limits[pair[0]] - coef[pair[0]]
                        if coef[pair[1]] - actual_dstep < lower_limits[pair[1]]:
                            actual_dstep = coef[pair[1]] - lower_limits[pair[1]]
                        new_coef = coef.copy()
                        new_coef[pair[0]] += actual_dstep
                        new_coef[pair[1]] -= actual_dstep
                        
                        ci = normal_data.apply(lambda row: mu_x(new_coef, row), axis=1)
                        new_quality = quality_indicator(ci)
                        if new_quality >= quality:
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
                # print(new_quality - quality)
                if new_quality > quality:
                    quality = new_quality
                    coef = new_coef
                elif abs(new_quality - quality) < limit_dquality:
                    reach  = min(
                        coef[pair[0]]-lower_limits[pair[0]],
                        upper_limits[pair[1]] - coef[pair[1]]
                    )
                    # print(("here",reach))
                    if reach >= 0:
                        actual_dstep = search_step(
                            coef, normal_data, mu_x, quality_indicator, 
                            quality, reach, pair[1], pair[0],
                            ray_depth, ray_search
                        )
                        if actual_dstep < dstep:
                            actual_dstep = dstep
                        if coef[pair[1]] + actual_dstep > upper_limits[pair[1]]:
                            actual_dstep = upper_limits[pair[1]] - coef[pair[1]]
                        if coef[pair[0]] - actual_dstep < lower_limits[pair[0]]:
                            actual_dstep = coef[pair[0]] - lower_limits[pair[0]]
                        new_coef = coef.copy()
                        new_coef[pair[0]] -= actual_dstep
                        new_coef[pair[1]] += actual_dstep
                        
                        ci = normal_data.apply(lambda row: mu_x(new_coef, row), axis=1)
                        new_quality = quality_indicator(ci)
                        if new_quality > quality:
                            quality = new_quality
                            coef = new_coef
                            updated = True
                    
            if limit_dquality <= abs(quality - old_quality):
                covered_directions = 0
            else:
                covered_directions += 1
        
            old_quality = quality
            pair = next(pairs_gen)
            n_step += 1
            
        return coef
    return Optimizer
