package Singleton;
import java.util.Random;

public class Test1 implements Runnable {

    private static Test1 instance = null;
    private final Random random = new Random();

    private Test1() {
    }

    public static Test1 getInstance() {
        if (instance == null) {
            instance = new Test1();
        }
        return instance;
    }

    @Override
    public void run() {
        int value = random.nextInt(100);
        System.out.println(Thread.currentThread().getName() + " generated: " + value);
    }
}