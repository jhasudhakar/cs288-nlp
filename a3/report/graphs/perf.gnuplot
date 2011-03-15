set xlabel 'Corpus Size (No. Sentences)'
set ylabel 'AER'
set term postscript enhanced 18
set output 'perf_aer.eps'
set key right top 
#set size 0.8,0.8
set size 0.7, 0.7
set yrange [0.0:1.0]
set logscale x

plot 'perf_heuristic.dat' using 3:7 title 'Heuristic' w linespoints, \
     'perf_model1.dat' using 3:7 title 'Model 1 Weird Soft EM' w linespoints, \
     'perf_hmm.dat' using 3:7 title 'HMM' w linespoints

set output 'perf_bleu.eps'
set key left top
set ylabel 'BLEU'
set yrange[1:30]
plot 'perf_heuristic.dat' using 3:9 title 'Heuristic' w linespoints, \
     'perf_model1.dat' using 3:9 title 'Model 1 Weird Soft EM' w linespoints, \
     'perf_hmm.dat' using 3:9 title 'HMM' w linespoints

