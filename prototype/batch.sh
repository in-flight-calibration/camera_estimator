run() { echo "### $*"; PYTHONPATH=pylib python3 solve.py "$@" 2>&1 | grep -E "FINAL|sd:|frames"; }
for s in 1 2 3 4; do run --seed $s; done
for k in 0 1 2; do run --part $k/3; done
run --hfov 40; run --hfov 100; run --pitch0 -20 --yaw0 10
run --huber 0.002; run --huber 0.02
run --frames 100000
run --rate 0,0.7 --frames 100000; run --rate 0.7,1.5 --frames 100000; run --rate 1.5,100 --frames 100000
run --max-range 1000
