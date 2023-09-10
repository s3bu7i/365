import android.os.Bundle
import android.text.TextUtils
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class RegisterActivity : AppCompatActivity() {

    private lateinit var usernameEditText: EditText
    private lateinit var passwordEditText: EditText
    private lateinit var confirmPasswordEditText: EditText
    private lateinit var registerButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var errorImageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_register)

        usernameEditText = findViewById(R.id.usernameEditText)
        passwordEditText = findViewById(R.id.passwordEditText)
        confirmPasswordEditText = findViewById(R.id.confirmPasswordEditText)
        registerButton = findViewById(R.id.registerButton)
        resultTextView = findViewById(R.id.resultTextView)
        errorImageView = findViewById(R.id.errorImageView)

        registerButton.setOnClickListener(View.OnClickListener {
            val username = usernameEditText.text.toString()
            val password = passwordEditText.text.toString()
            val confirmPassword = confirmPasswordEditText.text.toString()

            // Kullanıcı adı, parolaların ve onay parolasının boş olup olmadığını kontrol edin.
            if (TextUtils.isEmpty(username) || TextUtils.isEmpty(password) || TextUtils.isEmpty(confirmPassword)) {
                resultTextView.text = "Xanalar boş buraxıla bilməz"
                errorImageView.visibility = View.VISIBLE
                return@OnClickListener
            }

            // Parolanın minimum uzunluğunu kontrol edin (örneğin, 6 karakter).
            if (password.length < 6) {
                resultTextView.text = "Parol ən az 6 simvol olmalıdır"
                errorImageView.visibility = View.VISIBLE
                return@OnClickListener
            }

            // Parolanın ve onay parolasının eşleştiğini kontrol edin.
            if (password != confirmPassword) {
                resultTextView.text = "Parollar eşləşmir"
                errorImageView.visibility = View.VISIBLE
                return@OnClickListener
            }

            // Başarılı kayıt durumunda
            resultTextView.text = "Qeydiyyat uğurla başa çatdı"
            errorImageView.visibility = View.GONE
        })
    }
}
