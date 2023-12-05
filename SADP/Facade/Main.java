class Subsystem1 {
    public void operation1() {
        System.out.println("Subsystem1: Operation 1");
    }
}

class Subsystem2 {
    public void operation1() {
        System.out.println("Subsystem2: Operation 2");
    }
}

class Facade {
    private Subsystem1 subsystem1;
    private Subsystem2 subsystem2;

    public Facade() {
        this.subsystem1 = new Subsystem1();
        this.subsystem2 = new Subsystem2();
    }

    public void doSomething() {
        System.out.println("Facade: Doing something");
        subsystem1.operation1();
        subsystem2.operation1();
    }
}

public class Main {
    public static void main(String[] args) {
        Facade facade = new Facade();
        facade.doSomething();
    }
}
