import java.util.Scanner

fun main() {
    val input = Scanner(System.`in`)
    
    println("İki rəqəm daxil edin:")
    print("Rəqəm 1: ")
    val number1 = input.nextDouble()
    
    print("Rəqəm 2: ")
    val number2 = input.nextDouble()
    
    println("İşləmlər:")
    println("1. Toplama")
    println("2. Çıxma")
    println("3. Vurma")
    println("4. Bölmə")
    println("5. Qalıq göstərmə")
    
    print("İşləmi seçin (1-5): ")
    val operation = input.nextInt()
    
    when (operation) {
        1 -> {
            val result = number1 + number2
            println("$number1 + $number2 = $result")
        }
        2 -> {
            val result = number1 - number2
            println("$number1 - $number2 = $result")
        }
        3 -> {
            val result = number1 * number2
            println("$number1 * $number2 = $result")
        }
        4 -> {
            if (number2 != 0.0) {
                val result = number1 / number2
                println("$number1 / $number2 = $result")
            } else {
                println("0-a bölmə etmək olmaz.")
            }
        }
        5 -> {
            val result = number1 % number2
            println("$number1 % $number2 = $result")
        }
        else -> println("Yanlış əməliyyat seçdiniz.")
    }
}
