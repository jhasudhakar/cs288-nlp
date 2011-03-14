set xlabel 'Data Set Size'
set ylabel 'AER'
set term postscript enhanced 18
set output 'em.eps'
set key right top 
set size 0.7, 0.7
set yrange [0.20:0.70]
#set xrange [0.05:0.40]
set logscale x

plot 'em_weirdsoft.dat' using 3:7 title 'Weird Soft EM' w linespoints, \
     'em_soft.dat' using 3:7 title 'Soft EM' w linespoints, \
     'em_hard.dat' using 3:7 title 'Hard EM' w linespoints


