# CMU 16-745 Project: agile drone transportation


## Drone transportation task with optimal control

 | Standard Policy | Aggresive Policy |
 | --- | --- |
 | ![standard](assets/perodic_policy.gif) | ![aggressive](assets/swing_policy.gif) |
 | ![long horizon](assets/longhorizon.gif) | ![aggresive move](assets/aggresive_move.gif) |


 ## Usage

```bash
cd notebook
# dense solver
julia main.jl
# sparse solver (recommended)
julia main_sparse.jl
```