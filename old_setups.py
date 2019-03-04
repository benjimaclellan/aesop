"""
components = (
    {
     -1: Fiber(),
     1:AWG(),
     2:Fiber()
    })
adj = [(-1,1), (1,2)]
measurement_nodes = [2]
attributes = (
    {
     -1:[3],
     1:[2,0,np.pi/2],
     2:[3]
    })
"""


"""
components = (
    {
     -2:Fiber(),
     -1:Fiber(),
     0:PowerSplitter(),
     1:Fiber(),
     3:Fiber(),
     5:PowerSplitter(),
    }
    )
adj = [ (-2,-1), (-1,0), (0,1), (0,3),(3,5), (1,5) ]
measurement_nodes = [5]
attributes = (
    {
     -2:[0],
     -1:[0.0],
     1:[2],
     3:[0],
    })
"""



"""
components = (
    {
     'q':PhaseModulator(),
     's':AWG(),
     'd':AWG(),
     'f':AWG(),
     -2:FrequencySplitter(),
     -1:Fiber(),
     0:AWG(),
     'tt': AWG(),
     1:FrequencySplitter(),
#     2:FrequencySplitter(),
     3:AWG(),
     'dd':AWG(),
     4:Fiber(),
     'a':AWG(),
     6:FrequencySplitter(),
     7:Detector(),
     'benny':Detector()
    }
    )

adj = [ (1,'dd'), ('q',6), ('s','d'), ('d', 'f'), ('f', -2), (-2,-1), (-2,0), (0,1), (-1,1), (1,3), (1,4), (3,'a'), (4,6), ('a',6), (-2, 'tt'), ('tt', 6), (6,7), (7, 'benny')]
"""