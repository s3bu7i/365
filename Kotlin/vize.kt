import java.util.Scanner


fun main() {
    val input = Scanner(System.`in`)
    
    print("Vize qiymətini daxil edin: ")
    val vizeQiymeti = input.nextDouble()
    
    print("Final qiymətini daxil edin: ")
    val finalQiymeti = input.nextDouble()
    
    // Vize qiyməti ortalamanın 40 faizi, final qiyməti ortalamanın 60 faizi olacak
    val ortalama = (vizeQiymeti * 0.4) + (finalQiymeti * 0.6)
    
    // Ortalamaya əsasən harf notunu təyin et
    val harfNotu = when {
        ortalama >= 71 -> "A - əla"
        ortalama >= 61 -> "B - yaxşı"
        ortalama >= 51 -> "C - orta"
        else -> "F - kəsr"
    }
    
    // Ortalama və harf notunu istifadəçiyə göstər
    println("Ortalamanız: $ortalama")
    println("Harf notunuz: $harfNotu")
}

fun calculateWeeklySalary(hoursWorked: Double, hourlyRate: Double): Double {
    val normalHours = 40.0
    val overtimeRate = 1.5
    
    return if (hoursWorked <= normalHours) {
        hoursWorked * hourlyRate
    } else {
        val normalPay = normalHours * hourlyRate
        val overtimeHours = hoursWorked - normalHours
        val overtimePay = overtimeHours * hourlyRate * overtimeRate
        normalPay + overtimePay
    }
}

fun main() {
    val hoursWorked = 45.0 // İşçinin çalıştığı saatler
    val hourlyRate = 10.0 // Saat başına maaş oranı
    
    val weeklySalary = calculateWeeklySalary(hoursWorked, hourlyRate)
    
    println("Həftəlik maaş: $$weeklySalary")
}

