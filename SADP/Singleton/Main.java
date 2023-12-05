package Singleton;

public class Main {
    public static void main(String[] args) throws Exception {
        Test1 t = Test1.getInstance();
        Test1 t1 = Test1.getInstance();
        Test1 t2 = Test1.getInstance();
        Test1 t3 = Test1.getInstance();
        Test1 t4 = Test1.getInstance();

        Thread tt = new Thread(t);
        Thread tt2 = new Thread(t2);
        Thread tt3 = new Thread(t3);
        Thread tt4 = new Thread(t);
        Thread tt5 = new Thread(t);

        tt.start();
        tt2.start();
        tt3.start();
        tt4.start();
        tt5.start();
    }
}