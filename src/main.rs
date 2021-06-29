use ghost::phantom;
use std::rc::Rc;

fn main() {
    println!("Hello, world!");
    let a = Cont::<i32, i32>::pure(1);
    println!("a = {}", a.clone().eval_cont());
    let b = a.fmap(|x| x + 1);
    println!("b = {}", b.clone().run_cont(|x| x));
    let c = b.bind(|x: i32| Cont::pure(x + x));
    println!("c = {}", c.run_cont(|x| x));
    let a = Cont::<i32, i32>::pure(1);
    let b = a.fmap(|x| x + 1);
    println!("b = {}", b.clone().eval_cont());
    let a = Cont::<i32, i32>::pure(1);
    let b = a.bind(|x: i32| Cont::pure(x + x));
    println!("b = {}", b.eval_cont());
    let c = call_cc(|exit1| {
        Cont::pure(1)
            .bind(move |_: i32| exit1('a'))
            .bind(|_: i32| unimplemented!())
    });
    println!("c = {:#?}", c.eval_cont());
    // We can also do a nested exit
    let c = call_cc(|exit1| -> Cont<'_, char, char> {
        let exit1_clone = exit1.clone();
        call_cc(move |_: Rc<dyn Fn(char) -> Cont<'static, char, char>>| exit1('c'))
            .bind(move |_: char| exit1_clone('b'))
    });
    println!("c = {:#?}", c.eval_cont());
}

#[derive(Clone)]
pub struct Cont<'a, R, A> {
    //run: (a -> r) -> r
    pub run: Rc<dyn Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a>,
}

impl<'a, R, A> Cont<'a, R, A> {
    ///cont :: ((a -> r) -> r) -> Cont r a
    pub fn cont<F>(f: F) -> Cont<'a, R, A>
    where
        F: Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a,
    {
        Cont { run: Rc::new(f) }
    }
    pub fn fmap<B, F>(self, f: F) -> Cont<'a, R, B>
    where
        F: Fn(A) -> B + 'a + Clone,
        A: 'a,
        B: 'a,
        R: 'a,
    {
        //fmap f m = Cont $ \c -> run (c . f)
        //         = Cont (\c -> run (c . f))
        //         = Cont (\c -> run (\a -> c (f a)))
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(B) -> R + 'a>| {
                let f = f.clone();
                (self.run)(Rc::new(move |a: A| c(f(a))))
            }),
        }
    }
    pub fn pure(a: A) -> Cont<'a, R, A>
    where
        A: 'a + Clone,
    {
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(A) -> R>| c(a.clone())),
        }
    }
    pub fn bind<B, K>(self, k: K) -> Cont<'a, R, B>
    where
        K: Fn(A) -> Cont<'a, R, B> + 'a + Clone,
        A: 'a,
        R: 'a,
        B: 'a,
    {
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(B) -> R>| {
                let k = k.clone();
                (self.run)(Rc::new(move |a: A| (k(a).run)(c.clone())))
            }),
        }
    }
    pub fn run_cont<F>(self, f: F) -> R
    where
        F: Fn(A) -> R + 'a,
    {
        (self.run)(Rc::new(move |a: A| f(a)))
    }
}
impl<'a, R> Cont<'a, R, R> {
    pub fn eval_cont(self) -> R {
        self.run_cont(|r: R| r)
    }
}

pub fn call_cc<'a, A, B, R, F>(f: F) -> Cont<'a, R, A>
where
    F: for<'b> Fn(Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>) -> Cont<'a, R, A> + 'a,
    R: 'a + Clone,
    A: 'a + Clone,
{
    /* callCC f = ContT $ \ c -> runContT (f (\ x -> ContT $ \ _ -> c x)) c */
    let runit = call_cc_outer(f);
    Cont { run: runit }
}

fn call_cc_outer<'a, A, B, R, F>(f: F) -> Rc<dyn for<'b> Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a>
where
    F: for<'b> Fn(Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>) -> Cont<'a, R, A> + 'a,
    A: Clone + 'a,
    R: 'a,
{
    Rc::new(move |c| {
        let inner = call_cc_inner(c.clone());
        (f(inner).run)(c)
    })
}

fn call_cc_inner<'a, A, B, R>(c: Rc<dyn Fn(A) -> R + 'a>) -> Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>
where
    A: 'a + Clone,
    R: 'a,
{
    Rc::new(move |a: A| {
        let c = c.clone();
        Cont {
            run: Rc::new(move |_| c(a.clone())),
        }
    })
}

// http://blog.sigfpe.com/2008/12/mother-of-all-monads.html
// i x = cont (\fred -> x >>= fred)
// where cont f = Cont { run: f }
// i :: Monad m => m a -> Cont (m b) a
pub fn lift<'a, A, B, M, F>(m: This<M, A>) -> Cont<'a, This<M, B>, A>
where
    M: Monad<'a, A, B> + 'a + Clone,
    This<M, A>: Clone,
    A: Clone + 'a,
    B: Clone + 'a,
{
    Cont::cont(move |f| {
        let m = m.clone();
        M::bind(m, move |a: A| f(a))
    })
}

// Heavily borrowed from here: https://github.com/RustyYato/type-families
pub type This<T, A> = <T as Family<A>>::This;
pub trait Family<A> {
    type This;
}

pub trait Functor<'a, A, B>: Family<A> + Family<B> {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + 'a + Clone,
        A: 'a,
        B: 'a;
}

pub trait Pure<'a, A>: Family<A> {
    fn pure(value: A) -> This<Self, A>
    where
        A: 'a + Clone;
}

pub trait Applicative<'a, A, B>: Functor<'a, A, B> + Pure<'a, A> + Pure<'a, B> {
    fn apply<F>(a: This<Self, F>, b: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + 'a + Clone,
        Self: Applicative<'a, F, A>,
        A: 'a + Clone,
        B: 'a + Clone,
    {
        Self::lift_a2(move |q: F, r| q(r), a, b)
    }

    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + 'a + Clone,
        Self: Pure<'a, C>,
        Self: Family<C>,
        A: 'a + Clone,
        B: 'a + Clone,
        C: 'a + Clone;
}

pub trait Monad<'a, A, B>: Applicative<'a, A, B> {
    fn bind<F>(a: This<Self, A>, f: F) -> This<Self, B>
    where
        F: Fn(A) -> This<Self, B> + Clone + 'a;
    fn compose<F, G, C>(f: F, g: G, a: A) -> This<Self, C>
    where
        F: FnOnce(A) -> This<Self, B>,
        G: Fn(B) -> This<Self, C> + Clone + 'a,
        Self: Monad<'a, B, C>,
    {
        Self::bind(f(a), g)
    }
}

#[phantom]
#[derive(Clone, Copy)]
pub struct ContFamily<'a, R>;

impl<'a, R: 'a, A: 'a> Family<A> for ContFamily<'a, R> {
    type This = Cont<'a, R, A>;
}

impl<'a, R: 'a, A: 'a> Pure<'a, A> for ContFamily<'a, R> {
    fn pure(value: A) -> This<Self, A>
    where
        A: Clone + 'a,
    {
        Cont::pure(value)
    }
}

impl<'a, R: 'a, A: 'a, B: 'a> Functor<'a, A, B> for ContFamily<'a, R> {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + 'a + Clone,
    {
        this.fmap(f)
    }
}

impl<'a, R: 'a + Clone, A: 'a + Clone, B: 'a + Clone> Applicative<'a, A, B> for ContFamily<'a, R> {
    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + 'a + Clone,
        C: 'a + Clone,
    {
        a.bind(move |x: A| {
            let f = f.clone();
            let b = b.clone();
            b.bind(move |y: B| {
                let x = x.clone();
                Cont::pure(f(x, y))
            })
        })
    }
}

impl<'a, A, B, R> Monad<'a, A, B> for ContFamily<'a, R>
where
    R: Clone + 'a,
    A: Clone + 'a,
    B: Clone + 'a,
{
    fn bind<K>(a: This<Self, A>, k: K) -> This<Self, B>
    where
        K: Fn(A) -> Cont<'a, R, B> + 'a + Clone,
    {
        Cont::bind(a, k)
    }
}
