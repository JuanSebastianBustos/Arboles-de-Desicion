from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import logging
from werkzeug.exceptions import RequestEntityTooLarge, BadRequest

# Importar módulos de ML con manejo de errores
try:
    import RegresionLineal
except ImportError as e:
    print(f"Error importando RegresionLineal: {e}")
    RegresionLineal = None

try:
    import RegresionLogistica
except ImportError as e:
    print(f"Error importando RegresionLogistica: {e}")
    RegresionLogistica = None

try:
    import DecisionTrees
except ImportError as e:
    print(f"Error importando DecisionTrees: {e}")
    DecisionTrees = None

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datos de casos de uso (movidos a una función para mejor organización)
def get_casos_uso():
    return [
        {
            'id': 'entretenimiento',
            'titulo': 'Predicción de Abandono de Clientes (Churn) en Netflix',
            'industria': 'Streaming y Entretenimiento',
            'problema': 'El principal problema que se resuelve es la pérdida de ingresos debido a la cancelación de suscripciones. Adquirir un nuevo cliente es mucho más costoso que retener a uno existente. Por lo tanto, las empresas como Netflix necesitan identificar de manera proactiva a los usuarios que probablemente cancelarán su servicio (esto se conoce como "churn"). Al predecir quiénes están en riesgo, la empresa puede implementar estrategias de retención dirigidas para evitar que el cliente se vaya.',
            'algoritmo': 'Este es un problema de clasificación binaria (el cliente cancelará o no). Los algoritmos más comunes incluyen:\n- Regresión Logística\n- Máquinas de Soporte Vectorial (SVM)\n- Bosques Aleatorios (Random Forest), uno de los más efectivos.\nLos modelos se entrenan con datos históricos de los usuarios, como la frecuencia de inicio de sesión, el tiempo de visualización, los géneros preferidos y el historial de pagos.',
            'beneficios': '- Reducción de la Tasa de Abandono: Permite actuar antes de que el cliente se vaya, mejorando la retención.\n- Aumento de Ingresos: Retener más clientes asegura un flujo de ingresos más estable.\n- Optimización del Marketing: Las campañas de retención se dirigen solo a los clientes en riesgo, ahorrando costos.\n- Mejora de la Experiencia del Cliente: Las acciones de retención, como recomendaciones personalizadas, mejoran la satisfacción.',
            'empresa': 'Netflix es un ejemplo paradigmático. La compañía utiliza masivamente los datos de sus usuarios para alimentar sus algoritmos de recomendación y, de forma crucial, sus modelos de predicción de abandono para sostener su modelo de negocio.',
            'referencias': 'Ahmed, A. (2022). *Predicting Customer Churn in Python*. DataCamp. Recuperado de https://www.datacamp.com/tutorial/predicting-customer-churn-in-python\nJain, A. (2023, 10 de octubre). *How Netflix Uses Data Science to Enhance User Experience and Drive Business Growth*. Medium. Recuperado de https://medium.com/@anushka.jain992/how-netflix-uses-data-science-to-enhance-user-experience-and-drive-business-growth-4852c486411a'
        },
        {
            'id': 'salud',
            'titulo': 'Google/Verily — Detección de Retinopatía Diabética',
            'industria': 'Salud',
            'problema': 'La retinopatia diabetica es una de las principales causas de cegera evitable en el mundo. millones de pacientescon diabetes requieren exámenes oftalmologico regulares, pero existe una escasez global de especialistas que dificulta la deteccion temprana, sobre todo en zonas rurales o con pocos recursos medicos.',
            'algoritmo': 'Gloogle Research y Verily desarrollaron un sistema de redes neuronales convolucionales profundas (CNN) entrenado con mas de 120,000 imagenes de fondo de ojo, cuidadosamente etiquetadas por oftamologos certificados.El modelo clasifica las severidad de la retinopatia diabetica y el edema macular diabetico de forma automatica.',
            'beneficios': 'El algoritmo logró una sensibilidad de hasta 97% y una especificidad del 93%, comparable al desempeño de oftalmólogos expertos. Gracias a esto, se ha implementado en hospitales de India y Tailandia, logrando tamizajes masivos y diagnósticos tempranos, reduciendo el riesgo de ceguera y permitiendo que los especialistas se concentren en los casos más graves.',
            'empresa': 'Google Research y Verily Life Sciences, en colaboración con el Aravind Eye Hospital (India) y el Rajavithi Hospital (Tailandia).',
            'referencias': 'Google Research Asia-Pacific. (2024, octubre 17). Cómo la IA está haciendo que la atención para salvar la vista sea más accesible en entornos con recursos limitados. *Google Blog*. https://blog.google/around-the-globe/google-asia/arda-diabetic-retinopathy-india-thailand/\nAravind Eye Hospitals, Google LLC, & Verily Life Sciences LLC. (2023). Rendimiento de un algoritmo de aprendizaje profundo para la retinopatía diabética en India. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11923701/\nAndi, H. (2019, febrero 27). Google, Verily utilizar el aprendizaje automático para detectar la enfermedad ocular diabética. *Fierce Healthcare*. https://www.fiercehealthcare.com/tech/google-verily-develop-ai-tool-to-screen-for-diabetic-eye-disease'
        },
        {
            'id': 'educacion',
            'titulo': 'Modelo de predicción del éxito de Saber Pro (y trabajos relacionados).',
            'industria': 'Educación superior',
            'problema': 'Identificar qué factores (académicos y socioeconómicos) influyen en que un estudiante obtenga resultado por encima del promedio en la prueba Saber Pro – utilidad para políticas educativas y detección temprana de riesgo académico.',
            'algoritmo': 'Árboles de decisión (CART) como modelo supervisado de clasificación; estudios relacionados en Colombia también usan redes neuronales, regresión logística y bosques aleatorios según el alcance.',
            'beneficios': 'Identificación de variables influyentes, ej: puntaje en Saber 11 – especialmente la sección de Estudios Sociales –, características socioeconómicas y rasgos institucionales. Permite priorizar intervenciones educativas y diseñar estrategias de apoyo a estudiantes en riesgo. En estudios con grandes conjuntos de datos, ej: alrededor de 246.436 registros en análisis nacionales se obtuvieron modelos con rendimiento útil para análisis agregados y para decisiones institucionales.',
            'empresa': 'Trabajo académico de autores vinculados a Universidad EAFIT',
            'referencias': 'Pérez Bernal, G., Toro Villegas, L., & Toro, M. (2020). Saber Pro success prediction model using decision tree based learning. arXiv. https://arxiv.org/abs/2006.01322.\nBernal, G. P., Villegas, L. T., & Toro, M. (2020). Saber Pro success prediction model using decision tree based learning. ResearchGate. https://www.researchgate.net/publication/341851879_Saber_Pro_success_prediction_model_using_decision_tree_based_learning.\nVanegas-Ayala, S. C., Leal-Lara, D. D., & Barón-Velandia, J. (2022). Predicción rendimiento estudiantes pruebas Saber Pro en pandemia junto con las características socioeconómicas. Revista TIA – Tecnología, Investigación y Academia, 9(2), 5-16. Recuperado de https://revistas.udistrital.edu.co/index.php/tia/article/download/19446/18260/116618.'
        },
        {
            'id': 'servicios',
            'titulo': 'Detección de fraude en pagos (PayPal)',
            'industria': 'Servicios financieros',
            'problema': 'El fraude en transacciones digitales representa pérdidas millonaria para plataforma de pago y usuarios. Los sistemas tradicionales, basados en reglas fijas, no lograban adaptarse a la rapidez con la que evolucionan los patrones de fraude. Estos ocasionaba un gran número de falsos positivos, afectando la confianza de los clientes y aumentando los costos operativos.',
            'algoritmo': 'PayPal utiliza modelos de clasificación supervisada entrenados con millones de transacciones etiquetadas como "fraudulentas" o "legítimas". Entre los algoritmos aplicados destacan:\n- Gradient Boosting (para mejorar precisión en datos desbalanceados).\n- Redes neuronales profundas (Deep Learning) (para detectar patrones complejos y no lineales).\n- Ensembles de modelos (combinación de varios clasificadores para aumentar la robustez).\nEstos modelos permiten la detección en tiempo real de operaciones sospechosas, adaptándose dinámicamente a las nuevas estrategias de fraude.',
            'beneficios': '- Mayor precisión en la detección de fraude, reduciendo falsos positivos.\n- Prevención en tiempo real, protegiendo tanto a la empresa como a sus clientes.\n- Optimización de recursos, al reducir revisiones manuales.\n- Mejor experiencia de usuario, ya que menos transacciones legítimas son bloqueadas.',
            'empresa': 'PayPal Holdings, Inc. es una de las plataformas de pagos digitales más grandes del mundo, fundada en 1998 y con más de 430 millones de usuarios activos en 200 países. La compañía ha sido pionera en la integración de Machine Learning en sus sistemas de seguridad, conslidándose como un referente mundial en prevención de fraude financiero.',
            'referencias': 'Algotive. 5 ejemplos de Machine Learning que usas en tu día a día y no lo sabías:https://www.algotive.ai/es-mx/blog/5-ejemplos-de-machine-learning-que-usas-en-tu-dia-a-dia-y-no-lo-sabias#PayPal\nStripe. Cómo funciona el machine learning para prevenir el fraude en pagos: https://stripe.com/es/resources/more/how-machine-learning-works-for-payment-fraud-detection-and-prevention\nSEON. Machine Learning para detectar fraude: https://seon.io/es/recursos/machine-learning-para-detectar-fraude/\nInteractiveChaos.Gradient Boosting: https://interactivechaos.com/es/manual/tutorial-de-machine-learning/gradient-boosting\nIBM. ¿Qué es el deep learning?: https://www.ibm.com/es-es/think/topics/deep-learning\nBuiltIn. Modelos de conjunto ¿Qué son y cuándo utilizarlos?: https://builtin-com.translate.goog/machine-learning/ensemble-model\nWikipedia. PayPal: https://es.wikipedia.org/wiki/PayPal\nPayPal. History & facts: https://about.pypl.com/who-we-are/history-and-facts/default.aspx'
        }
    ]

# Funciones de utilidad para validación
def validate_numeric_input(value, min_val=None, max_val=None, field_name="Campo"):
    """Valida entrada numérica con rangos opcionales."""
    try:
        num_value = float(value)
        if min_val is not None and num_value < min_val:
            return False, f"{field_name} debe ser mayor o igual a {min_val}"
        if max_val is not None and num_value > max_val:
            return False, f"{field_name} debe ser menor o igual a {max_val}"
        return True, num_value
    except (ValueError, TypeError):
        return False, f"{field_name} debe ser un número válido"

def check_dataset_exists():
    """Verifica si el dataset de regresión logística existe."""
    return os.path.exists('Datasets/data.csv')

# Manejadores de errores
@app.errorhandler(404)
def not_found_error(error):
    return render_template('pagina_error.html', 
                         error_title="Página no encontrada",
                         error_message="La página que buscas no existe.",
                         error_code=404), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Server Error: {error}')
    return render_template('pagina_error.html',
                         error_title="Error interno del servidor",
                         error_message="Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo.",
                         error_code=500), 500

@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return render_template('pagina_error.html',
                         error_title="Archivo demasiado grande",
                         error_message="El archivo enviado supera el límite de tamaño permitido.",
                         error_code=413), 413

# Rutas principales
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caso1')
def caso1():
    casos_uso = get_casos_uso()
    caso = casos_uso[0]
    return render_template('caso1.html', caso=caso)

@app.route('/caso2')
def caso2():
    casos_uso = get_casos_uso()
    caso = casos_uso[1]
    return render_template('caso2.html', caso=caso)

@app.route('/caso3')
def caso3():
    casos_uso = get_casos_uso()
    caso = casos_uso[2]
    return render_template('caso3.html', caso=caso)

@app.route('/caso4')
def caso4():
    casos_uso = get_casos_uso()
    caso = casos_uso[3]
    return render_template('caso4.html', caso=caso)

# Rutas de Regresión Lineal

@app.route('/rl_conceptos')
def rl_conceptos():
    return render_template('rl_conceptos.html')

@app.route('/rl_practico', methods=['GET', 'POST'])
def rl_practico():
    if RegresionLineal is None:
        flash('Error: Módulo de Regresión Lineal no disponible.', 'error')
        return redirect(url_for('index'))
    
    plot_url = None
    result = None
    autos = None
    transporte = None
    errors = {}

    try:
        # Generar el gráfico
        plot_url = RegresionLineal.PlotGraph()
    except Exception as e:
        logger.error(f'Error generando gráfico: {e}')
        flash('Error al generar la visualización. Por favor, inténtalo de nuevo.', 'error')

    if request.method == 'POST':
        try:
            # Validar entrada de autos
            autos_input = request.form.get('autos')
            valid_autos, autos_result = validate_numeric_input(
                autos_input, min_val=0, max_val=10000, field_name="Número de autos"
            )
            
            if not valid_autos:
                errors['autos'] = autos_result
            else:
                autos = autos_result

            # Validar entrada de transporte público
            transporte_input = request.form.get('transporte')
            valid_transporte, transporte_result = validate_numeric_input(
                transporte_input, min_val=0, max_val=100, field_name="Uso de transporte público"
            )
            
            if not valid_transporte:
                errors['transporte'] = transporte_result
            else:
                transporte = transporte_result

            # Si no hay errores, realizar predicción
            if not errors:
                result = RegresionLineal.PredecirContaminacion(autos, transporte)
                flash('Predicción realizada exitosamente.', 'success')
            else:
                for field, error in errors.items():
                    flash(error, 'error')
                    
        except Exception as e:
            logger.error(f'Error en predicción de regresión lineal: {e}')
            flash('Error al procesar la predicción. Verifica los datos ingresados.', 'error')

    return render_template(
        'rl_practico.html',
        plot_url=plot_url,
        result=result,
        autos=autos,
        transporte=transporte,
        errors=errors
    )

# Rutas de Regresión Logística

@app.route('/logistica_conceptos')
def logistica_conceptos():
    return render_template('logistica_conceptos.html')

@app.route('/logistica_practico', methods=['GET', 'POST'])
def logistica_practico():
    if RegresionLogistica is None:
        flash('Error: Módulo de Regresión Logística no disponible.', 'error')
        return redirect(url_for('index'))

    if not check_dataset_exists():
        flash('Error: Dataset no encontrado. Contacta al administrador.', 'error')
        return redirect(url_for('index'))

    try:
        # Obtener métricas de evaluación
        accuracy, report, matriz_url = RegresionLogistica.evaluate()
        
        # Filtrar el reporte para excluir métricas no numéricas
        report_clean = {
            clase: metrics for clase, metrics in report.items() 
            if isinstance(metrics, dict) and clase in ['0', '1']
        }
        
    except Exception as e:
        logger.error(f'Error evaluando modelo de regresión logística: {e}')
        flash('Error al cargar las métricas del modelo.', 'error')
        return redirect(url_for('logistica_conceptos'))

    # Variables para predicción
    prediction = None
    probability = None
    features = None
    errors = {}

    if request.method == 'POST':
        try:
            # Validar horas de práctica
            horas_input = request.form.get('horas_practica')
            valid_horas, horas_result = validate_numeric_input(
                horas_input, min_val=0, max_val=168, field_name="Horas de práctica"
            )
            
            if not valid_horas:
                errors['horas_practica'] = horas_result

            # Validar experiencia previa
            experiencia_input = request.form.get('experiencia_previa')
            if experiencia_input not in ['0', '1']:
                errors['experiencia_previa'] = "Debe seleccionar si tiene experiencia previa"
            
            # Validar nota media
            nota_input = request.form.get('nota_media')
            valid_nota, nota_result = validate_numeric_input(
                nota_input, min_val=0, max_val=5, field_name="Nota media"
            )
            
            if not valid_nota:
                errors['nota_media'] = nota_result

            # Validar asistencia
            asistencia_input = request.form.get('asistencia')
            valid_asistencia, asistencia_result = validate_numeric_input(
                asistencia_input, min_val=0, max_val=100, field_name="Asistencia"
            )
            
            if not valid_asistencia:
                errors['asistencia'] = asistencia_result

            # Si no hay errores, realizar predicción
            if not errors:
                features = [horas_result, int(experiencia_input), nota_result, asistencia_result]
                prediction, probability = RegresionLogistica.predict_label(features)
                flash('Predicción realizada exitosamente.', 'success')
            else:
                for field, error in errors.items():
                    flash(error, 'error')

        except Exception as e:
            logger.error(f'Error en predicción de regresión logística: {e}')
            flash('Error al procesar la predicción. Verifica los datos ingresados.', 'error')

    return render_template(
        'logistica_practico.html',
        accuracy=accuracy,
        report=report_clean,
        matriz_url=matriz_url,
        prediction=prediction,
        probability=probability,
        features=features,
        errors=errors
    )

# Rutas de Árboles de Decisión

@app.route('/dt_conceptos')
def dt_conceptos():
    return render_template('dt_conceptos.html')

#ruta para la parte práctica de árboles de decisión


@app.route('/dt_practico', methods=['GET', 'POST'])
def dt_practico():
    if DecisionTrees is None:
        flash('Error: Módulo de Árboles de Decisión no disponible.', 'error')
        return redirect(url_for('index'))

    try:
        # Siempre evaluamos el modelo para tener las métricas base
        resultados_eval = DecisionTrees.evaluate()
        
        # Variables para la predicción
        prediction_result = None
        form_data = None
        

        

        if request.method == 'POST':
            # Recoger datos del formulario
            promedio = float(request.form['promedio_academico'])
            ingresos = int(request.form['ingresos_familiares'])
            personas = int(request.form['numero_personas_hogar'])
            tipo_inst = int(request.form['tipo_institucion']) # Esto ya es 0 para Publica, 1 para Privada

            
            form_data = {
                "promedio": promedio,
                "ingresos": ingresos,
                "personas": personas,
                "tipo_inst": tipo_inst
            }

            # Llamamos a la función pasando los valores directamente en el orden correcto
            pred_label, pred_prob = DecisionTrees.predict_label(promedio, ingresos, personas, tipo_inst)
            
            prediction_result = {
                "label": pred_label,
                "probability": f"{pred_prob:.4f}"
            }

        # Renderizar la plantilla con los resultados de la evaluación y la predicción (si existe)
        return render_template(
            'dt_practico.html',
            accuracy=resultados_eval['accuracy'],
            reporte=resultados_eval['reporte_dict'], # Pasamos el diccionario
            matriz_url=resultados_eval['matriz_path'],
            prediction=prediction_result,
            form_data=form_data 
        )
        
    except Exception as e:
        # AÑADIMOS ESTAS LÍNEAS PARA VER EL ERROR EN LA TERMINAL
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!!!!!!!!!!! ERROR ATRAPADO !!!!!!!!!!!!!!")
        print(f"TIPO DE ERROR: {type(e).__name__}")
        print(f"MENSAJE DE ERROR: {e}")
        import traceback
        traceback.print_exc() # Esto imprime el historial completo del error
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        logging.error(f'Error en la ruta de árboles de decisión: {e}')
        flash(f'Ocurrió un error inesperado: {e}', 'error')
        return redirect(url_for('dt_conceptos'))



# API endpoints para AJAX (opcional)
@app.route('/api/validate_field', methods=['POST'])
def validate_field():
    """Endpoint para validación en tiempo real de campos."""
    data = request.get_json()
    field_name = data.get('field')
    value = data.get('value')
    
    # Definir reglas de validación por campo
    validation_rules = {
        'horas_practica': {'min': 0, 'max': 168, 'name': 'Horas de práctica'},
        'nota_media': {'min': 0, 'max': 5, 'name': 'Nota media'},
        'asistencia': {'min': 0, 'max': 100, 'name': 'Asistencia'},
        'autos': {'min': 0, 'max': 10000, 'name': 'Número de autos'},
        'transporte': {'min': 0, 'max': 100, 'name': 'Uso de transporte público'},
        'edad': {'min': 17, 'max': 25, 'name': 'Edad'},
        'promedio_secundaria': {'min': 2.0, 'max': 5.0, 'name': 'Promedio de secundaria'},
        'ingresos_familiares': {'min': 500000, 'max': 5000000, 'name': 'Ingresos familiares'},
        'horas_estudio': {'min': 0, 'max': 40, 'name': 'Horas de estudio'}
    }
    
    if field_name in validation_rules:
        rules = validation_rules[field_name]
        is_valid, result = validate_numeric_input(
            value, rules['min'], rules['max'], rules['name']
        )
        
        if is_valid:
            return jsonify({'valid': True, 'message': 'Válido'})
        else:
            return jsonify({'valid': False, 'message': result})
    
    return jsonify({'valid': False, 'message': 'Campo no reconocido'})

# Funciones de contexto para templates
@app.context_processor
def inject_globals():
    """Inyecta variables globales a todos los templates."""
    return {
        'app_name': 'ML Supervisado',
        'app_version': '2.0',
        'current_year': 2025
    }

if __name__ == '__main__':
    # Verificar que los módulos y archivos necesarios existen
    missing_components = []
    
    if RegresionLineal is None:
        missing_components.append("RegresionLineal.py")
    
    if RegresionLogistica is None:
        missing_components.append("RegresionLogistica.py")
        
    if DecisionTrees is None:
        missing_components.append("DecisionTrees.py")
        
    if not check_dataset_exists():
        missing_components.append("Datasets/data.csv")
    
    if missing_components:
        logger.warning(f"Componentes faltantes: {', '.join(missing_components)}")
        print(f"⚠️  Advertencia: Faltan componentes: {', '.join(missing_components)}")
        print("La aplicación funcionará con funcionalidad limitada.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)