package edu.berkeley.nlp.assignments.assign1.student;

public class Utils {
   
   public static final long mask = -1L >>> 32;

   public static long pack(int left, int right) {
     return ((long)left << 32) | ((long)right & mask);
   }

   public static int left(long v) {
     return (int)(v >>> 32);
   }

   public static int right(long v) {
     return (int)(v & mask);
   }

   
   public static void reportMemoryUsage() {
      System.gc(); System.gc(); System.gc(); System.gc();
      long totalMem = Runtime.getRuntime().totalMemory();
      long freeMem = Runtime.getRuntime().freeMemory();
      System.out.println("Memory usage is " + bytesToString(totalMem - freeMem));
   }
   
   private static String bytesToString(long b) {
      double mb = (double) b / (1024 * 1024);
      if (mb >= 1) return mb >= 10 ? (int) mb + "M" : round(mb, 1) + "M";
      double kb = (double) b / (1024);
      if (kb >= 1) return kb >= 10 ? (int) kb + "K" : round(kb, 1) + "K";
      return b + "";
   }
   
   private static double round(double x, int numPlaces) {
      double scale = Math.pow(10, numPlaces);
      return Math.round(x * scale) / scale;
   }
}
