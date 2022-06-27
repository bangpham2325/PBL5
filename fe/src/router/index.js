import Vue from 'vue'
import VueRouter from 'vue-router'
import HomeView from '../views/HomeView.vue'

Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'introduce',
    meta: {layout: 'blank'},
    component: () => import('../views/Introduce.vue')
  },
  {
    path: '/about',
    name: 'about',
    component: () => import('../views/AboutView.vue')
  },
  {
    path: '/home',
    name: 'home',
    component: HomeView
  },
  {
    path: '/classification',
    name: 'classification',
    component: () => import('../views/Classification.vue')
  },
  {
    path: '*',
    name: 'error',
    meta: {layout: 'blank'},
    component: () => import('../views/Error.vue')
  },

]

const router = new VueRouter({
  mode: 'history',
  routes
})

export default router
