import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import vuetify from './plugins/vuetify'
import Default from './layouts/Default.vue'
import Blank from './layouts/Blank.vue'
import { library } from '@fortawesome/fontawesome-svg-core'
import { fas } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { BootstrapVue, IconsPlugin } from 'bootstrap-vue'
import 'vuetify/dist/vuetify.min.css'
import Embed from 'v-video-embed'
import {Chart} from 'chart.js'
import Chartkick from 'vue-chartkick' 
import VueChartkick from 'vue-chartkick'
import VCalendar from 'v-calendar';
import Calendar from 'v-calendar/lib/components/calendar.umd'
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
import VueResource from 'vue-resource'

import datePicker from 'vue-bootstrap-datetimepicker';
import 'pc-bootstrap4-datetimepicker/build/css/bootstrap-datetimepicker.css';

library.add(fas)
Vue.component('font-awesome-icon', FontAwesomeIcon)


Vue.component('default-layout',Default)
Vue.component('blank-layout',Blank)
Vue.component('calendar', Calendar)
Vue.component('date-picker', datePicker)
Vue.use(Embed);
Vue.use(Chartkick.use(Chart));
Vue.use(VueChartkick)
Vue.use(VCalendar)
Vue.use(BootstrapVue)
Vue.use(IconsPlugin)
Vue.use(VueResource)

Vue.config.productionTip = false

new Vue({
  router,
  store,
  vuetify,
  render: h => h(App),
}).$mount('#app')
