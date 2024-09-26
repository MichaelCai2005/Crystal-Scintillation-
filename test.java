public class test{

    public static void main(String[] args) {
        int n = 4;  // Example value for n
        int x = 6; // Example value for x
        int d = 7;  // Example value for d

        int result = mystery(n, x, d);
        System.out.println("Result: " + result); // Output the result
    }

    public static int mystery(int n, int x, int d) {
        if (n == 1) {
            return x;
        }

        return d + mystery(n - 1, x, d);
    }
}