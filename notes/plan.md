

function retunrs:



Dict per configuration:
    C_tilde_DD - Non-Maxwell Capacitance
    C_DG - Gate-dot capacitance matrix
    tc - Tunnel couplings
    v_offset - Offset voltage for the gate
    "cut"::
       '001-100':
        x_voltage - [x0,x1,N] linspace, taking into account the sizes
        y_voltage - [y0,y1,N] linspace, taking into account the sizes
        
    


C_tilde_DD  -> array(batch_size, Nd, Nd)
C_DG -> array(batch_size, Nd, Ng)
tc -> array(batch_size, Nd, Nd)
v_offset -> array(batch_size, Ng)


x_voltage -> array(batch_size, Ncuts, 3)
y_voltage -> array(batch_size, Ncuts, 3)
lut_cuts -> ['100-100':0, '001-110':1, ... ]



data['cut]['voltage']

    













