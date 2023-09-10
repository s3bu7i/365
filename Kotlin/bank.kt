fun main() {
    val input = Scanner(System.`in`)
    
    print("Adınızı daxil edin: ")
    val ad = input.nextLine()
    
    print("Rəqəmi daxil edin: ")
    val reqem = input.nextInt()
    
    for (i in 1..reqem) {
        println(ad)
    }
}

fun main() {
    val input = Scanner(System.`in`)
    
    print("Bir rəqəm daxil edin: ")
    val reqem = input.nextInt()
    
    when {
        reqem == 0 -> println("Proqram dayandırıldı.")
        reqem % 2 == 0 -> println("Daxil etdiyiniz rəqəm cütdür.")
        else -> println("Daxil etdiyiniz rəqəm təkdir.")
    }
}

fun main() {
    val input = Scanner(System.`in`)
    
    print("Adınızı daxil edin: ")
    val ad = input.nextLine()
    
    print("Kartın başlangıç büdcəsini daxil edin (AZN): ")
    var butce = input.nextDouble()
    
    while (true) {
        println("Zəhmət olmasa bir əməliyyat seçin:")
        println("1. Kartın əməliyyat tarixini göstər")
        println("2. Kartın balansını göstər")
        println("3. Pul daxil et")
        println("4. Pul çıxar")
        println("5. Çıxış")
        
        val emeliyyat = input.nextInt()
        
        when (emeliyyat) {
            1 -> println("$ad adlı şəxsın kartın əməliyyat tarixi yoxdur.")
            2 -> println("$ad adlı şəxsın kartının balansı: $butce AZN")
            3 -> {
                print("Daxil etmək istədiyiniz məbləği daxil edin (AZN): ")
                val mebleq = input.nextDouble()
                butce += mebleq
                println("$mebleq AZN kartınıza əlavə edildi. Yeni balans: $butce AZN")
            }
            4 -> {
                print("Çıxarmaq istədiyiniz məbləği daxil edin (AZN): ")
                val mebleq = input.nextDouble()
                if (mebleq <= butce) {
                    butce -= mebleq
                    println("$mebleq AZN kartınızdan çıxarıldı. Yeni balans: $butce AZN")
                } else {
                    println("Balansınızdan çıxarmaq üçün kifayət qədər pulunuz yoxdur.")
                }
            }
            5 -> {
                println("Proqram dayandırıldı.")
                break
            }
            else -> println("Yalnış əməliyyat nömrəsi. Zəhmət olmasa yenidən seçin.")
        }
    }
}

